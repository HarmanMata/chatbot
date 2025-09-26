import streamlit as st
import google.generativeai as genai
import requests
import urllib.parse
from datetime import datetime
import os
from dotenv import load_dotenv

# -----------------------------
# RAG Imports
# -----------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
# FIX: Use langchain_google_genai instead of community for the standard Google embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# -----------------------------
# API Keys (FIXED & RENAMED for LangChain compatibility)
# -----------------------------
# Load environment variables from .env file
load_dotenv()

# We look for both for backwards compatibility, but store in the preferred name
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# FIX: Set the environment variable if loaded, which is often needed for LangChain/Google SDK
if GEMINI_API_KEY:
    os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Please add it to your environment variables or a .env file.")
if not SERPAPI_KEY:
    st.warning("SERPAPI_KEY is not set. Web search will be disabled.")


# Configure the generative AI model
# Only configure if the key is present to avoid errors during initial load
if GEMINI_API_KEY:
    # Use the Google SDK for the main chat model
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash") # Stable model is used
else:
    model = None

# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "AI answer"
# Check if the FAISS index file exists to determine RAG readiness
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = os.path.exists("faiss_index")

# -----------------------------
# Helper Functions
# -----------------------------
def real_time_search(query):
    """Performs a real-time web search using SerpAPI."""
    if not SERPAPI_KEY:
        return "⚠️ SERPAPI_KEY not configured. Web search disabled."
    try:
        q = urllib.parse.quote_plus(query)
        # Using Google engine for general search
        url = f"https://serpapi.com/search.json?q={q}&engine=google&api_key={SERPAPI_KEY}"
        r = requests.get(url, timeout=10).json()
        snippets = [it.get("snippet", "") for it in r.get("organic_results", [])[:5] if it.get("snippet")]
        return "\n\n".join(snippets) if snippets else "⚠️ No live results found."
    except Exception as e:
        return f"⚠️ Web search failed: {e}"

def ai_answer_stream(user_input):
    """Generates a response from the AI model with chat history context."""
    if model is None:
        return iter([type('obj', (object,), {'text': "Model not ready due to missing API Key."})()])
        
    # Create a simple history string for context
    history_for_prompt = st.session_state.chat_history[-5:]
    history_string = "\n".join([f"{r}: {t}" for r, t, ts in history_for_prompt])
    
    # Construct a robust prompt including context for conversation
    prompt = f"The following is a conversation history:\n{history_string}\n\nUser: {user_input}\nAI:"
    return model.generate_content(prompt, stream=True)

def summarize_web_results_stream(results, query):
    """Summarizes web search results."""
    if results.startswith("⚠️"):
        # Create a mock chunk object if there was an error, so the stream loop handles it
        yield type('obj', (object,), {'text': results})()
        return
    if model is None:
        yield type('obj', (object,), {'text': "Model not ready to summarize results."})()
        return
        
    prompt = f"Summarize this for the question: {query}\n\n{results}"
    response_generator = model.generate_content(prompt, stream=True)
    for chunk in response_generator:
        yield chunk

# -----------------------------
# RAG Helper Functions
# -----------------------------

def prepare_documents(pdf_docs, api_key):
    """Reads PDFs, chunks text, and creates/saves the FAISS vector store."""
    if not api_key:
        raise ValueError("API Key is missing for embedding model. Cannot create knowledge base.")
        
    raw_text = "".join(page.extract_text() for pdf in pdf_docs for page in PdfReader(pdf).pages if page.extract_text())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)
    
    # FIX: The key parameter for LangChain's GoogleGenerativeAIEmbeddings is 'google_api_key'
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key) 
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def create_rag_prompt(user_question, api_key):
    """Loads the knowledge base, retrieves context, and augments the prompt."""
    if not api_key:
        raise ValueError("API Key is missing for embedding model. Cannot retrieve context.")
        
    # FIX: The key parameter for LangChain's GoogleGenerativeAIEmbeddings is 'google_api_key'
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    # The allow_dangerous_deserialization is required for loading a FAISS index from disk
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = db.similarity_search(user_question, k=4)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"""
    Answer the user's question ONLY based on the following context. If the answer is not in the context, clearly state that you don't know or that the information is not in the documents.
    
    Context:
    ---
    {context}
    ---
    
    Question: {user_question}
    """

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS for a Clean, Simple UI
# -----------------------------
st.markdown("""
<style>
    body {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #0d1117;
    }
    
    .stChatMessage {
        margin: 0 auto;
        max-width: 800px;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    
    .stChatMessage[data-testid="stChatMessage"] .st-emotion-cache-1c7er9e {
        background-color: #1a2533;
        color: #e6edf3;
    }

    .stChatMessage[data-testid="stChatMessage"] .st-emotion-cache-43u8q9 {
        background-color: #111a22;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    
    .st-emotion-cache-16t2a7b input, .st-emotion-cache-16t2a7b textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }
    
    .st-emotion-cache-16t2a7b .st-emotion-cache-u885f8 {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

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
# RAG UI (Sidebar)
# -----------------------------
with st.sidebar:
    st.header("Upload Private Documents (RAG)")
    pdf_docs = st.file_uploader(
        "Upload your PDF Files here (e.g., policy docs, reports)",
        accept_multiple_files=True,
        type=['pdf']
    )
    if st.button("Create Knowledge Base"):
        if pdf_docs:
            if not GEMINI_API_KEY:
                st.error("Please set your Gemini API key (GEMINI_API_KEY) to create the knowledge base.")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        # Passing the key to the RAG preparation function
                        prepare_documents(pdf_docs, GEMINI_API_KEY)
                        st.session_state.rag_ready = True
                        st.success("Knowledge Base Ready! You can now use 'RAG with my documents' mode.")
                    except ValueError as ve:
                         # Catching the explicit ValueError added in the helper function
                        st.error(f"Configuration Error: {ve}")
                        st.session_state.rag_ready = False
                    except Exception as e:
                        # Updated error message to guide the user on the correct environment variable name
                        if "Authentication Failed" in str(e) or "API Key not found" in str(e) or "400 Client Error" in str(e):
                            st.error(f"Error processing documents: Authentication Failed. Please ensure you have set the **GEMINI_API_KEY** environment variable correctly and that the key is valid.")
                        else:
                            st.error(f"Error processing documents: {e}")
                        st.session_state.rag_ready = False
        else:
            st.error("Please upload at least one PDF file.")


# -----------------------------
# UI Layout
# -----------------------------
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("<h2 style='text-align:left; color:#e6edf3;'> AI Chatbot</h2>", unsafe_allow_html=True)
with col2:
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        # FIX: Replace deprecated st.experimental_rerun() with st.rerun()
        st.rerun() 

st.markdown("<hr style='border:1px solid #30363d; margin-top: 0;'>", unsafe_allow_html=True)

mode_options = ["AI answer", "Web + summarize"]
if st.session_state.rag_ready:
    mode_options.append("RAG with my documents")

# Ensure the selected mode is still valid if the knowledge base was deleted or not created
if st.session_state.mode not in mode_options:
    st.session_state.mode = "AI answer"

st.session_state.mode = st.selectbox(
    "Response mode:",
    mode_options,
    help="Choose 'Web + summarize' for real-time information. Choose 'RAG with my documents' for answers from your uploaded files."
)

# Display existing chat history
for role, text, ts in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# -----------------------------
# Input Box Logic
# -----------------------------
if prompt := st.chat_input("Type your question..."):
    ts = datetime.now().strftime("%H:%M")
    
    # 1. Add user prompt to history and display it
    st.session_state.chat_history.append(("User", prompt, ts))
    with st.chat_message("User"):
        st.write(prompt)

    # 2. Get AI response based on mode
    with st.chat_message("AI"):
        if not model:
            # Handle uninitialized model gracefully
            st.error("Model not initialized. Please ensure your Gemini API key is set.")
            response_generator = iter([type('obj', (object,), {'text': "Model not ready. Please check API Key configuration."})()])
        
        elif st.session_state.mode == "AI answer":
            response_generator = ai_answer_stream(prompt)
            
        elif st.session_state.mode == "Web + summarize":
            with st.spinner('Searching the web...'):
                results = real_time_search(prompt)
            response_generator = summarize_web_results_stream(results, prompt)
            
        elif st.session_state.mode == "RAG with my documents":
            if not st.session_state.rag_ready:
                st.error("Please create the knowledge base in the sidebar first by uploading documents and clicking 'Create Knowledge Base'.")
                full_response = "Error: RAG knowledge base not ready."
                response_generator = iter([type('obj', (object,), {'text': full_response})()])
            else:
                with st.spinner('Searching your documents and generating answer...'):
                    try:
                        # Passing the key to the RAG prompt function
                        rag_prompt = create_rag_prompt(prompt, GEMINI_API_KEY)
                        # Use the configured model
                        response_generator = model.generate_content(rag_prompt, stream=True)
                    except Exception as e:
                        st.error(f"Error accessing knowledge base or generating content: {e}")
                        full_response = "Error: Could not retrieve document context or generate a response."
                        response_generator = iter([type('obj', (object,), {'text': full_response})()])

        # 3. Stream the response
        full_response = ""
        placeholder = st.empty()
        for chunk in response_generator:
            if hasattr(chunk, 'text'):
                full_response += chunk.text
                placeholder.markdown(full_response + "▌") # Show typing cursor
            
        placeholder.markdown(full_response) # Final response without cursor
    
    # 4. Update chat history with the final AI response
    st.session_state.chat_history.append(("AI", full_response, ts))
    # FIX: Replace deprecated st.experimental_rerun() with st.rerun()
    st.rerun()
