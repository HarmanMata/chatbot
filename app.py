import streamlit as st
import google.generativeai as genai
import requests
import urllib.parse
from datetime import datetime

# -----------------------------
# API Keys
# -----------------------------
GENAI_API_KEY = "AIzaSyDLaa1Ru5klYJgPVbWWZwRia__2NUFo2Gs"      # <-- replace with your Gemini API key
SERPAPI_KEY = "fe0dabbd93be11974bed64d0b151446d123608e504ec55f2a9b98cf62319497f"      # <-- replace with your SerpAPI key (optional for web search)

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # (role, text, timestamp)
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
        snippets = [it.get("snippet","") for it in r.get("organic_results", [])[:5] if it.get("snippet")]
        return "\n\n".join(snippets) if snippets else "⚠️ No live results found."
    except Exception as e:
        return f"⚠️ Web search failed: {e}"

def ai_answer_stream(user_input):
    history_for_prompt = st.session_state.chat_history[-5:]
    history_string = "\n".join([f"{r}: {t}" for r,t,ts in history_for_prompt])
    
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
# Dark Theme CSS
# -----------------------------
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.markdown("""
<style>
body { background-color:#0d1117; color:#e6edf3; font-family:Inter, sans-serif; }
.block-container { padding:20px; }
.chat-box { max-width:800px; margin:auto; }
.msg { padding:12px 16px; margin:8px 0; border-radius:10px; max-width:80%; line-height:1.5; }
.user { background:#1f6feb; color:white; margin-left:auto; text-align:right; }
.ai { background:#161b22; border:1px solid #30363d; }
.meta { font-size:11px; color:#8b949e; margin-top:4px; }
input, textarea, select { background:#161b22 !important; color:#e6edf3 !important; border-radius:6px !important; border:1px solid #30363d !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Heading
# -----------------------------
st.markdown("<h2 style='text-align:center; color:#e6edf3;'> AI Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #30363d;'>", unsafe_allow_html=True)

# -----------------------------
# Render Chat History
# -----------------------------
for role, text, ts in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# -----------------------------
# Mode Selector
# -----------------------------
st.session_state.mode = st.selectbox("Response mode:", ["AI answer", "Web + summarize"])

# -----------------------------
# Input Box (Enter to Send)
# -----------------------------
if prompt := st.chat_input("Type your question..."):
    ts = datetime.now().strftime("%H:%M")
    
    # Append the user's message to the session history
    st.session_state.chat_history.append(("User", prompt, ts))
    
    # Display the user's message immediately
    with st.chat_message("User"):
        st.write(prompt)

    # Create a new chat message container for the AI's response
    with st.chat_message("AI"):
        
        # Determine the response generator based on the selected mode
        if st.session_state.mode == "AI answer":
            response_generator = ai_answer_stream(prompt)
        else:
            results = real_time_search(prompt)
            response_generator = summarize_web_results_stream(results, prompt)

        # Stream the response from the generator
        full_response = ""
        placeholder = st.empty()
        for chunk in response_generator:
            if hasattr(chunk, 'text'):
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")  # Add a typing cursor for visual effect
        
        # Once streaming is done, remove the cursor and update the final response
        placeholder.markdown(full_response)
    
    # Append the complete, final response to the session history
    st.session_state.chat_history.append(("AI", full_response, ts))
    st.rerun()

# -----------------------------
# Clear Chat
# -----------------------------
if st.button("Clear chat"):
    st.session_state.chat_history = []
    st.rerun()