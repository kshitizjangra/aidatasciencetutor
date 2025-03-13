import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.retry import Retry

# Set environment variable for DNS resolver
os.environ["GRPC_DNS_RESOLVER"] = "native"

# Initialize the Streamlit app
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")
st.title("AI Data Science Tutor")

# -----------------------
# 1. Session State Setup
# -----------------------
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store conversation history manually
if "loading" not in st.session_state:
    st.session_state.loading = False

# -----------------------
# 2. Custom Retry Policy
# -----------------------
retry_policy = Retry(
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    deadline=900.0
)

# -----------------------
# 3. Sidebar: API Key + Controls
# -----------------------
with st.sidebar:
    st.title("Configuration Settings")
    
    with st.container():
        st.markdown("<h3 style='margin-bottom: 10px;'>Configure the API before using this tool.</h3>", unsafe_allow_html=True)
        api_key = st.text_input(
            "Enter your Google Gemini API Key",
            placeholder="Paste your API key here.",
            key="api_key_input"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if api_key:
                try:
                    st.session_state.llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        google_api_key=api_key,
                        temperature=0.7,
                        retry=retry_policy
                    )
                    st.session_state.chat_history.clear()  # Reset history when new API key is set
                    st.success("API Key configured successfully")
                except Exception as e:
                    st.error(f"Invalid API Key or authentication error: {e}")
            else:
                st.session_state.llm = None

    with st.container():
        st.markdown("<h3 style='margin-bottom: 10px;'>Conversation Controls</h3>", unsafe_allow_html=True)
        if st.button("Clear Conversation History"):
            st.session_state.chat_history.clear()
            st.success("Conversation history cleared!")

# -----------------------
# 4. Chat UI
# -----------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("<h3 style='margin-bottom: 20px;'>Conversation</h3>", unsafe_allow_html=True)
    
    # Display chat history immediately
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:  
            st.markdown(f"**Tutor:** {message['content']}")

    # Chat input
    user_input = st.chat_input("Ask your data science question...")

# -----------------------
# 5. Handle Input & Response (Instant Reply Fix)
# -----------------------
if user_input and st.session_state.llm:
    with st.spinner("Thinking..."):
        # Add user message to the chat history immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Prepare messages for AI
        messages = [{"role": "system", "content": "You are a helpful AI Data Science Tutor. Keep responses technical and concise."}]
        
        # Add chat history to maintain context
        for msg in st.session_state.chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        try:
            # Get AI response
            response = st.session_state.llm.invoke(messages)
            ai_response = response.content

            # Add AI response immediately
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

            # Refresh page to show updates
            st.rerun()
        
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

elif user_input:
    st.warning("Please enter your API Key first")
