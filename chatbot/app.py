import streamlit as st
import google.generativeai as genai
import requests

# -----------------------------
# Gemini Setup
# -----------------------------
genai.configure(api_key="AIzaSyDLaa1Ru5klYJgPVbWWZwRia__2NUFo2Gs")  # Replace with your API key
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Initialize Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "context" not in st.session_state:
    st.session_state.context = ""

# -----------------------------
# DuckDuckGo Search
# -----------------------------
def duckduckgo_search(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        response = requests.get(url)
        data = response.json()
        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", []):
            if "Text" in topic:
                results.append(topic["Text"])
        return "\n\n".join(results[:3]) if results else "⚠️ No results found."
    except:
        return "⚠️ Search failed."

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Chat", page_icon="🤖", layout="wide")
st.markdown(
    "<h2 style='text-align: center; font-family: sans-serif;'>🤖 AI Chatbot</h2>",
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar - Chat History
# -----------------------------
st.sidebar.title("📜 Chat History")
if st.sidebar.button("🗑 Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.context = ""
if st.session_state.chat_history:
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.sidebar.expander(f"Q{i+1}: {q}"):
            st.write(a)
else:
    st.sidebar.info("No past chats yet.")

# -----------------------------
# Main UI
# -----------------------------
user_input = st.text_input("💬 Type your question here:")

col1, col2 = st.columns(2)
with col1:
    ask_ai = st.button("✨ Ask AI")
with col2:
    web_search = st.button("🌐 Search Web")

# -----------------------------
# Handle Responses
# -----------------------------
latest_answer = None

if ask_ai and user_input:
    try:
        # Combine context + new user input
        prompt = st.session_state.context + "\nUser: " + user_input + "\nAI:"
        response = model.generate_content(prompt)
        latest_answer = response.text.strip()

        # Update chat history and context
        st.session_state.chat_history.append((user_input, latest_answer))
        st.session_state.context += f"\nUser: {user_input}\nAI: {latest_answer}"
    except Exception as e:
        latest_answer = f"⚠️ AI Error: {e}"

if web_search and user_input:
    latest_answer = duckduckgo_search(user_input)
    st.session_state.chat_history.append((f"(Web Search) {user_input}", latest_answer))

# -----------------------------
# Display Latest Q&A
# -----------------------------
if latest_answer:
    st.markdown("---")
    st.write(latest_answer)
