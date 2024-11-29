# Importing necessary libraries for the system
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
import docx
from typing import Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import requests
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for database collections
university_collection = None
living_expenses_collection = None
employment_collection = None
chroma_client = None
embedding_function = None

# Location to city mapping for weather
LOCATION_TO_CITY = {
    "Northeast": "New York",
    "Southeast": "Miami",
    "Midwest": "Chicago",
    "Southwest": "Houston",
    "West Coast": "Los Angeles"
}

# User data file path
USER_DATA_FILE = "user_data.json"

# Chatbot usage tips
CHATBOT_TIPS = """
### 💡 Tips for Using COMPASS

1. **Ask Specific Questions**
   - About university programs
   - About living costs in different locations
   - About job prospects in your field
   - About weather conditions

2. **Get Recommendations**
   - Click "Top 3 Recommendations" for personalized suggestions
   - Ask follow-up questions about specific universities
   - Inquire about admission requirements

3. **Explore Details**
   - Ask about specific universities
   - Request cost breakdowns
   - Learn about campus life
   - Get weather information

4. **Example Questions**
   - "Tell me more about Georgia Tech's program"
   - "What are the living costs in Atlanta?"
   - "How's the job market for data science in the Northeast?"
   - "Compare the weather between Boston and Miami"
"""

def load_user_data() -> dict:
    """Load user data from JSON file."""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user data: {str(e)}")
    return {}

def save_user_data(data: dict):
    """Save user data to JSON file."""
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving user data: {str(e)}")

def authenticate_user(username: str) -> bool:
    """Authenticate user and load their data."""
    users = load_user_data()
    
    # Initialize session state for new user
    if username not in users:
        users[username] = {
            "preferences": None,
            "chat_history": [],
            "last_recommendations": None,
            "created_at": datetime.now().isoformat()
        }
        save_user_data(users)
    
    # Load user data into session state
    st.session_state.current_user = username
    st.session_state.user_data = users[username]
    st.session_state.authenticated = True
    
    return True

def save_user_preferences(username: str, preferences: dict):
    """Save user preferences."""
    users = load_user_data()
    if username in users:
        users[username]["preferences"] = preferences
        save_user_data(users)
        st.session_state.user_data = users[username]

def save_chat_history(username: str, chat_history: list):
    """Save chat history for a user."""
    users = load_user_data()
    if username in users:
        users[username]["chat_history"] = chat_history
        save_user_data(users)

def save_last_recommendations(username: str, recommendations: str):
    """Save last recommendations for context."""
    users = load_user_data()
    if username in users:
        users[username]["last_recommendations"] = recommendations
        save_user_data(users)
        st.session_state.user_data = users[username]

def load_word_document(file_path: str) -> str:
    """Load content from a Word document."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        st.error(f"Error loading Word document: {str(e)}")
        return ""

def initialize_chromadb():
    """Initialize ChromaDB with OpenAI embeddings."""
    global university_collection, living_expenses_collection, employment_collection
    global chroma_client, embedding_function
    
    try:
        # Setup OpenAI embedding function
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=st.secrets["open-key"],
            model_name="text-embedding-ada-002"
        )
        
        # Connect to existing ChromaDB instance using PersistentClient
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get existing collections
        university_collection = chroma_client.get_collection(
            name="university_info",
            embedding_function=embedding_function
        )
        
        living_expenses_collection = chroma_client.get_collection(
            name="living_expenses",
            embedding_function=embedding_function
        )
        
        employment_collection = chroma_client.get_collection(
            name="employment_projections",
            embedding_function=embedding_function
        )
        
        logger.info("Successfully connected to existing ChromaDB")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {str(e)}")
        st.error(f"Failed to initialize database: {str(e)}")
        return False

def load_initial_data():
    """Load initial data into ChromaDB collections."""
    try:
        # Load datasets
        living_expenses_df = pd.read_csv(os.path.join("data", "Avg_Living_Expenses.csv"))
        employment_df = pd.read_csv(os.path.join("data", "Employment_Projections.csv"))
        university_text = load_word_document(os.path.join("data", "University_Data.docx"))
        
        # Process living expenses
        logger.info("Processing living expenses data...")
        for idx, row in living_expenses_df.iterrows():
            content = (
                f"State: {row['State']}\n"
                f"Cost of Living Index: {row['Index']}\n"
                f"Grocery: {row['Grocery']}\n"
                f"Housing: {row['Housing']}\n"
                f"Utilities: {row['Utilities']}\n"
                f"Transportation: {row['Transportation']}\n"
                f"Health: {row['Health']}\n"
                f"Miscellaneous: {row['Misc.']}"
            )
            living_expenses_collection.add(
                documents=[content],
                metadatas=[{
                    "state": row["State"].strip(),
                    "type": "living_expenses",
                    "index": float(row["Index"])
                }],
                ids=[f"living_expenses_{idx}"]
            )
        
        # Process employment projections
        logger.info("Processing employment projections data...")
        for idx, row in employment_df.iterrows():
            content = (
                f"Occupation: {row['Occupation Title']}\n"
                f"Employment 2023: {row['Employment 2023']}\n"
                f"Growth Rate: {row['Employment Percent Change, 2023-2033']}%\n"
                f"Annual Openings: {row['Occupational Openings, 2023-2033 Annual Average']}\n"
                f"Median Wage: ${row['Median Annual Wage 2023']}\n"
                f"Required Education: {row['Typical Entry-Level Education']}"
            )
            employment_collection.add(
                documents=[content],
                metadatas=[{
                    "occupation": row["Occupation Title"],
                    "type": "employment",
                    "median_wage": float(row["Median Annual Wage 2023"])
                }],
                ids=[f"employment_{idx}"]
            )
        
        # Process university data
        logger.info("Processing university data...")
        chunk_size = 1000
        chunks = [university_text[i:i + chunk_size] for i in range(0, len(university_text), chunk_size)]
        
        for idx, chunk in enumerate(chunks):
            # Add some basic text preprocessing
            chunk = chunk.strip()
            if not chunk:  # Skip empty chunks
                continue
                
            university_collection.add(
                documents=[chunk],
                metadatas=[{
                    "chunk_id": idx,
                    "type": "university",
                    "length": len(chunk)
                }],
                ids=[f"university_{idx}"]
            )
        
        logger.info("Successfully loaded initial data into ChromaDB")
        
    except Exception as e:
        logger.error(f"Error loading initial data: {str(e)}")
        raise e

def reset_chromadb():
    """Reset ChromaDB collections (useful for testing)."""
    global chroma_client
    try:
        if chroma_client:
            for collection_name in ["university_info", "living_expenses", "employment_projections"]:
                try:
                    chroma_client.delete_collection(collection_name)
                    logger.info(f"Deleted collection: {collection_name}")
                except:
                    pass
        return initialize_chromadb()
    except Exception as e:
        logger.error(f"Error resetting ChromaDB: {str(e)}")
        return False
    
def get_living_expenses(state: str) -> str:
    """Query living expenses information."""
    try:
        results = living_expenses_collection.query(
            query_texts=[state],
            n_results=1
        )
        return results["documents"][0][0] if results["documents"][0] else "No information found."
    except Exception as e:
        logger.error(f"Error in get_living_expenses: {str(e)}")
        return f"Error retrieving living expenses: {str(e)}"

def get_job_market_trends(field: str) -> str:
    """Query job market trends."""
    try:
        results = employment_collection.query(
            query_texts=[field],
            n_results=3
        )
        return "\n\n".join(results["documents"][0])
    except Exception as e:
        logger.error(f"Error in get_job_market_trends: {str(e)}")
        return f"Error retrieving job market trends: {str(e)}"

def get_university_info(query: str) -> str:
    """Query university information."""
    try:
        results = university_collection.query(
            query_texts=[query],
            n_results=3
        )
        return "\n\n".join(results["documents"][0])
    except Exception as e:
        logger.error(f"Error in get_university_info: {str(e)}")
        return f"Error retrieving university information: {str(e)}"

def get_weather_info(location: str) -> str:
    """Get weather information for a location's major city."""
    try:
        city = LOCATION_TO_CITY.get(location, location)
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{city},US",
            "appid": st.secrets["open-weather"],
            "units": "imperial"
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return (f"Current weather in {city}: "
               f"Temperature: {data['main']['temp']}°F, "
               f"Feels like: {data['main']['feels_like']}°F, "
               f"Humidity: {data['main']['humidity']}%, "
               f"Conditions: {data['weather'][0]['description']}")
    except Exception as e:
        logger.error(f"Error in get_weather_info: {str(e)}")
        return f"Error retrieving weather information: {str(e)}"

def get_top_recommendations() -> str:
    """Generate top 3 university recommendations based on user preferences."""
    try:
        preferences = st.session_state.user_data["preferences"]
        if not preferences:
            return "Please set your preferences first to get personalized recommendations."

        prompt = f"""Generate top 3 university recommendations based on these preferences:
        - Field of Study: {preferences['field_of_study']}
        - Budget Range: ${preferences['budget_min']}-${preferences['budget_max']}
        - Preferred Locations: {', '.join(preferences['preferred_locations'])}
        - Weather Preference: {preferences['weather_preference']}

        Format each recommendation as:
        [Number]. [University Name]
        * Key strengths in {preferences['field_of_study']}
        * Cost and aid highlights
        * Location and weather notes
        * Brief distinguishing features

        Keep each university description detailed but concise. Include specific programs, costs, and unique features."""

        client = OpenAI(api_key=st.secrets["open-key"])
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a university advisor providing detailed, specific recommendations."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        recommendations = response.choices[0].message.content
        save_last_recommendations(st.session_state.current_user, recommendations)
        return recommendations

    except Exception as e:
        logger.error(f"Error generating top recommendations: {str(e)}")
        return "Error generating recommendations. Please try again."

def get_recommendations(query: str) -> str:
    """Generate recommendations based on user query and preferences."""
    try:
        # Check if query is about a previously recommended university
        last_recommendations = st.session_state.user_data.get("last_recommendations", "")
        context = []
        
        if last_recommendations and any(university.lower() in query.lower() 
                                      for university in ["georgia tech", "unc", "vanderbilt", "chapel hill"]):
            context.append(("Previous Recommendations:", last_recommendations))
        
        # Get university information
        uni_info = get_university_info(query)
        if uni_info:
            context.append(("University Information:", uni_info))
        
        # Get living expenses if state is mentioned
        if any(word in query.lower() for word in ["state", "cost", "living", "expensive"]):
            expenses_info = get_living_expenses(query)
            if expenses_info:
                context.append(("Living Expenses:", expenses_info))
        
        # Get job market trends if career/job is mentioned
        if any(word in query.lower() for word in ["job", "career", "employment", "salary", "work"]):
            job_info = get_job_market_trends(st.session_state.user_data["preferences"]["field_of_study"])
            if job_info:
                context.append(("Job Market Trends:", job_info))
        
        # Get weather for relevant locations
        if st.session_state.user_data["preferences"]["preferred_locations"]:
            location_weather = []
            for location in st.session_state.user_data["preferences"]["preferred_locations"]:
                weather_info = get_weather_info(location)
                if not weather_info.startswith("Error"):
                    location_weather.append(weather_info)
            if location_weather:
                context.append(("Weather Information:", "\n".join(location_weather)))
        
        # Prepare prompt with context and user preferences
        context_text = "\n\n".join([f"{title}\n{info}" for title, info in context])
        preferences = st.session_state.user_data["preferences"]
        
        prompt = f"""As a university advisor, help with this query: {query}

Context:
{context_text}

Student Profile:
- Field: {preferences['field_of_study']}
- Budget: ${preferences['budget_min']}-${preferences['budget_max']}
- Locations: {', '.join(preferences['preferred_locations'])}
- Weather: {preferences['weather_preference']}

Previous conversation context (if relevant):
{last_recommendations}

Provide a detailed, specific response focusing on the query. If the query is about a specific university from the previous recommendations, reference that information."""

        # Generate response using OpenAI
        client = OpenAI(api_key=st.secrets["open-key"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": ("You are a helpful university advisor for international students. "
                              "When asked about specific universities from previous recommendations, "
                              "maintain consistency with that information while adding more details. "
                              "Provide specific, actionable insights.")
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return "I apologize, but I encountered an error. Please try asking a more specific question."

def initialize_session_state():
    """Initialize all session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'show_preferences' not in st.session_state:
        st.session_state.show_preferences = False

def show_login_page():
    """Display the login page."""
    st.header("👋 Welcome to COMPASS")
    
    with st.form("login_form"):
        username = st.text_input("Enter your name")
        submitted = st.form_submit_button("Login")
        
        if submitted and username:
            if authenticate_user(username):
                st.success(f"Welcome, {username}!")
                if not st.session_state.user_data.get("preferences"):
                    st.info("Please set your preferences to continue.")
                st.rerun()

def show_preferences_form(existing_preferences=None):
    """Display the preferences setup/edit form."""
    with st.form("preferences_form"):
        # Field of Study (text input)
        field_of_study = st.text_input(
            "Field of Study",
            value=existing_preferences.get("field_of_study", "") if existing_preferences else "",
            help="Enter your intended field of study"
        )
        
        # Budget Range
        default_min = existing_preferences.get("budget_min", 20000) if existing_preferences else 20000
        default_max = existing_preferences.get("budget_max", 50000) if existing_preferences else 50000
        budget_range = st.slider(
            "Budget Range (USD/Year)",
            0, 100000, (default_min, default_max),
            help="Select your annual budget range"
        )
        
        # Preferred Locations
        default_locations = existing_preferences.get("preferred_locations", []) if existing_preferences else []
        preferred_locations = st.multiselect(
            "Preferred Locations",
            ["Northeast", "Southeast", "Midwest", "Southwest", "West Coast"],
            default=default_locations,
            help="Select your preferred locations"
        )
        
        # Weather Preference
        default_weather = existing_preferences.get("weather_preference", "Moderate") if existing_preferences else "Moderate"
        weather_preference = st.select_slider(
            "Weather Preference",
            options=["Cold", "Moderate", "Warm", "Hot"],
            value=default_weather,
            help="Select your preferred weather type"
        )
        
        submitted = st.form_submit_button("Save Preferences")
        
        if submitted:
            if not field_of_study or not preferred_locations:
                st.error("Please fill in all required fields.")
                return False
            
            preferences = {
                "field_of_study": field_of_study,
                "budget_min": budget_range[0],
                "budget_max": budget_range[1],
                "preferred_locations": preferred_locations,
                "weather_preference": weather_preference
            }
            
            save_user_preferences(st.session_state.current_user, preferences)
            st.success("✅ Preferences saved successfully!")
            return True
    return False

def show_sidebar():
    """Display sidebar with tips and preferences."""
    with st.sidebar:
        st.title("🎓 COMPASS Guide")
        
        # Show current user
        st.write(f"👤 Logged in as: {st.session_state.current_user}")
        
        # Current Preferences Section
        st.header("📋 Your Preferences")
        prefs = st.session_state.user_data.get("preferences", {})
        if prefs:
            st.write(f"**Field of Study:** {prefs.get('field_of_study')}")
            st.write(f"**Budget:** ${prefs.get('budget_min'):,} - ${prefs.get('budget_max'):,}")
            st.write(f"**Locations:** {', '.join(prefs.get('preferred_locations', []))}")
            st.write(f"**Weather:** {prefs.get('weather_preference')}")
            
            if st.button("✏️ Edit Preferences"):
                st.session_state.show_preferences = True
        
        # Tips Section
        st.markdown(CHATBOT_TIPS)
        
        # Logout button at the bottom
        if st.button("🚪 Logout", key="logout_sidebar"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def show_chat_interface():
    """Display the main chat interface."""
    # Show preferences edit form if requested
    if st.session_state.show_preferences:
        st.header("✏️ Edit Preferences")
        if show_preferences_form(st.session_state.user_data.get("preferences")):
            st.session_state.show_preferences = False
            st.rerun()
        return
    
    st.header("💬 Chat with COMPASS")
    
    # Top recommendations and clear chat buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌟 Top 3 Recommendations"):
            recommendations = get_top_recommendations()
            st.session_state.user_data["chat_history"].append({
                "role": "assistant",
                "content": recommendations
            })
            save_chat_history(st.session_state.current_user, 
                            st.session_state.user_data["chat_history"])
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.user_data["chat_history"] = []
            save_chat_history(st.session_state.current_user, [])
            st.rerun()

    # Display chat history
    for message in st.session_state.user_data["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about universities, programs, costs, or job prospects..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add to chat history
        st.session_state.user_data["chat_history"].append({
            "role": "user",
            "content": prompt
        })
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_recommendations(prompt)
                st.write(response)
                
                # Add to chat history
                st.session_state.user_data["chat_history"].append({
                    "role": "assistant",
                    "content": response
                })
        
        # Save updated chat history
        save_chat_history(
            st.session_state.current_user,
            st.session_state.user_data["chat_history"]
        )

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="COMPASS - University Recommendation System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize ChromaDB if not already initialized
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            if initialize_chromadb():
                st.session_state.initialized = True
            else:
                st.error("Failed to initialize the system. Please refresh the page.")
                return
    
    # Display appropriate page based on authentication state
    if not st.session_state.authenticated:
        show_login_page()
    else:
        # Show sidebar
        show_sidebar()
        
        # Show preferences page if preferences not set
        if not st.session_state.user_data.get("preferences"):
            show_preferences_form()
        else:
            show_chat_interface()

if __name__ == "__main__":
    main()