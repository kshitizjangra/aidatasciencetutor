import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.retry import Retry

# Set environment variable for DNS resolver
os.environ["GRPC_DNS_RESOLVER"] = "native"

# Initialize the Streamlit app
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")
st.title("AI Data Science Tutor")

# Initializing the Session state 
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loading" not in st.session_state:
    st.session_state.loading = False

# Custom retry policy
retry_policy = Retry(
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    deadline=900.0
)

# Inputting the API Key in this project using (Google Gemini) in the sidebar
with st.sidebar:
    st.title("Configuration Settings")
    
    with st.container():
        st.markdown("<h3 style='margin-bottom: 10px;'>Configure the API before using this tool..</h3>", unsafe_allow_html=True)
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
                    st.session_state.memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                    st.success("API Key configured successfully")
                except Exception as e:
                    st.error(f"Invalid API Key or authentication error: {e}")
            else:
                st.session_state.llm = None

    with st.container():
        st.markdown("<h3 style='margin-bottom: 10px;'>Conversation Controls</h3>", unsafe_allow_html=True)
        if st.button("Clear Conversation History"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.success("Conversation history cleared!")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("<h3 style='margin-bottom: 20px;'>Conversation</h3>", unsafe_allow_html=True)
    
    # Display chat history along with the style enhancements
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div style='display: flex; margin-bottom: 15px;'>
                    <div class='user-avatar'></div>
                    <div class='chat-message-user'>
                        {message["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='display: flex; margin-bottom: 15px;'>
                    <div class='assistant-avatar'></div>
                    <div class='chat-message-assistant'>
                        {message["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # User input along with the character counter
    user_input = st.chat_input("Ask your data science question...")
    if user_input:
        st.markdown(f"<p style='font-size: 12px; color: #666;'>Characters: {len(user_input)}</p>", unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='margin-bottom: 20px;'>Conversation Details</h3>", unsafe_allow_html=True)
    st.markdown(f"**Total Messages:** {len(st.session_state.chat_history)}")
    st.markdown(f"**Model Temperature:** 0.7")
    st.markdown(f"**Current API Status:** {'Connected' if st.session_state.llm else 'Disconnected'}")

# Handle user input and AI response
if user_input and st.session_state.llm:
    st.session_state.loading = True
    progress_bar = st.progress(0)
    progress_bar.progress(25)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Create message history with the system prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI Data Science Tutor. Answer only data science related questions. Keep responses technical and concise."},
        *st.session_state.memory.load_memory_variables({})["chat_history"]
    ]
    
    # Add current user message
    messages.append({"role": "user", "content": user_input})
    
    try:
        # Get the response from the AI model
        response = st.session_state.llm.invoke(messages)
        
        # Updating the progress bar
        progress_bar.progress(75)
        
        # Adding the assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response.content})
        
        # Updating memory
        st.session_state.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )
        
        # Final progress update
        progress_bar.progress(100)
        st.session_state.loading = False
        
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        st.session_state.loading = False
elif user_input:
    st.warning("Please enter your API Key first")

# Display the loading indicator
if st.session_state.loading:
    with st.spinner("Thinking..."):
        st.empty()