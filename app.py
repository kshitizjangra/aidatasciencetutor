import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.retry import Retry

# Set environment variable for DNS resolver
os.environ["GRPC_DNS_RESOLVER"] = "native"

# Initializing the Streamlit app
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")
st.title("AI Data Science Tutor")

# Setting up the Session State
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "loading" not in st.session_state:
    st.session_state.loading = False

# Custom Retry Policy
retry_policy = Retry(
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    deadline=900.0
)

# API Key and Controls in the Sidebar
with st.sidebar:
    st.title("Configuration Settings")
    
    with st.container():
        st.markdown("<h3 style='margin-bottom: 10px;'>Configure the API before using this tool..</h3>", unsafe_allow_html=True)
        api_key = st.text_input(
            "Enter your Google Gemini API Key",
            placeholder="Paste your API key here..",
            key="api_key_input"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if api_key:
                try:
                    st.session_state.llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro",
                        google_api_key=api_key,
                        temperature=0.7,
                        retry=retry_policy
                    )
                    # Clear memory whenever a new API key is set
                    st.session_state.conversation_memory.clear()
                    st.success("API Key configured successfully")
                except Exception as e:
                    st.error(f"Invalid API Key or authentication error: {e}")
            else:
                st.session_state.llm = None

    with st.container():
        st.markdown("<h3 style='margin-bottom: 10px;'>Conversation Controls</h3>", unsafe_allow_html=True)
        if st.button("Clear Conversation History"):
            st.session_state.conversation_memory.clear()
            st.success("Conversation history cleared successfully")

# Chat UI
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("<h3 style='margin-bottom: 20px;'>Conversation</h3>", unsafe_allow_html=True)
    
    # Displaying the chat history from memory
    for msg in st.session_state.conversation_memory:
        # Identify the role of each message (Human, AI, System)
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**Tutor:** {msg['content']}")
        elif msg["role"] == "system":
            st.markdown(f"_System_: {msg['content']}")

    # Chat input
    user_input = st.chat_input("Ask your data science question..")

# Input & Response Handling
if user_input and st.session_state.llm:
    with st.spinner("Thinking..."):
        # Add the user's message to memory
        st.session_state.conversation_memory.append({"role": "user", "content": user_input})

        # Building the final message list, starting with a system prompt
        system_prompt = {
            "role": "system",
            "content": "You are a helpful AI Data Science Tutor. Keep responses technical and concise. " \
                       "Only answer questions related to data science, machine learning, statistics, " \
                       "data analysis, and related technical topics. If asked about other topics, " \
                       "politely inform the user that you specialize in data science and suggest " \
                       "they ask about data-related topics."
        }
        
        messages = [system_prompt]
        messages.extend(st.session_state.conversation_memory)

        # Invoke Gemini
        try:
            response = st.session_state.llm.invoke(messages)
            ai_response = response.content

            # Check if response is about data science
            if "data science" in ai_response.lower() or "machine learning" in ai_response.lower() or "statistics" in ai_response.lower():
                # Adding the AI's response back into the memory
                st.session_state.conversation_memory.append({"role": "assistant", "content": ai_response})
            else:
                # Create a polite response about being a data science tutor
                polite_response = "I'm specialized in data science topics. Could you please ask a question related to data science, " \
                                 "machine learning, statistics, or data analysis?"
                st.session_state.conversation_memory.append({"role": "assistant", "content": polite_response})

            # Rerun to show the updated conversation
            st.rerun()

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

elif user_input:
    st.warning("Please enter your API Key first")
