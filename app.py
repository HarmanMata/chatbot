import streamlit as st
import google.generativeai as genai
import requests
import urllib.parse
from datetime import datetime

# -----------------------------
# API Keys
# -----------------------------
GENAI_API_KEY = "AIzaSyDLaa1Ru5klYJgPVbWWZwRia__2NUFo2Gs"
# SERPAPI_KEY = "fe0dabbd93be11974bed64d0b151446d123608e504ec55f2a9b98cf62319497f"

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "AI answer"

# -----------------------------
# Helper Functions
# -----------------------------
def real_time_search(query):
    if not SERPAPI_KEY or SERPAPI_KEY.startswith("YOUR"):
        return "⚠️ SERPAPI_KEY not configured. Web search disabled."
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://serpapi.com/search.json?q={q}&engine=google&api_key={SERPAPI_KEY}"
        r = requests.get(url, timeout=10).json()
        snippets = [it.get("snippet", "") for it in r.get("organic_results", [])[:5] if it.get("snippet")]
        return "\n\n".join(snippets) if snippets else "⚠️ No live results found."
    except Exception as e:
        return f"⚠️ Web search failed: {e}"

def ai_answer_stream(user_input):
    history_for_prompt = st.session_state.chat_history[-5:]
    history_string = "\n".join([f"{r}: {t}" for r, t, ts in history_for_prompt])
    prompt = f"{history_string}\nUser: {user_input}\nAI:"
    return model.generate_content(prompt, stream=True)

def summarize_web_results_stream(results, query):
    if results.startswith("⚠️"):
        yield type('obj', (object,), {'text': results})()
        return
    prompt = f"Summarize this for the question: {query}\n\n{results}"
    response_generator = model.generate_content(prompt, stream=True)
    for chunk in response_generator:
        yield chunk

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS for a Clean, Simple UI
# -----------------------------
st.markdown("""
<style>
    /* Main container and body */
    body {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #0d1117;
    }
    
    /* Center the chat messages and give them a max width */
    .stChatMessage {
        margin: 0 auto;
        max-width: 800px;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="stChatMessage"] .st-emotion-cache-1c7er9e {
        background-color: #1a2533;
        color: #e6edf3;
    }

    /* AI message styling */
    .stChatMessage[data-testid="stChatMessage"] .st-emotion-cache-43u8q9 {
        background-color: #111a22;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    
    /* Input field styling */
    .st-emotion-cache-16t2a7b input, .st-emotion-cache-16t2a7b textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }
    
    /* Selectbox styling */
    .st-emotion-cache-16t2a7b .st-emotion-cache-u885f8 {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* Clear chat button styling */
    .stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stButton > button:hover {
        background-color: #30363d;
        color: #e6edf3;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI Layout
# -----------------------------

# Header with clear button
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("<h2 style='text-align:left; color:#e6edf3;'> AI Chatbot</h2>", unsafe_allow_html=True)
with col2:
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("<hr style='border:1px solid #30363d; margin-top: 0;'>", unsafe_allow_html=True)

# Mode selector and chat history display
st.session_state.mode = st.selectbox(
    "Response mode:", 
    ["AI answer", "Web + summarize"],
    help="Choose 'Web + summarize' for real-time information from the web."
)

for role, text, ts in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# -----------------------------
# Input Box Logic
# -----------------------------
if prompt := st.chat_input("Type your question..."):
    ts = datetime.now().strftime("%H:%M")
    
    st.session_state.chat_history.append(("User", prompt, ts))
    with st.chat_message("User"):
        st.write(prompt)

    with st.chat_message("AI"):
        if st.session_state.mode == "AI answer":
            response_generator = ai_answer_stream(prompt)
        else:
            with st.spinner('Searching the web...'):
                results = real_time_search(prompt)
            response_generator = summarize_web_results_stream(results, prompt)

        full_response = ""
        placeholder = st.empty()
        for chunk in response_generator:
            if hasattr(chunk, 'text'):
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
    
    st.session_state.chat_history.append(("AI", full_response, ts))
    st.rerun()