import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.retry import Retry

# For conversation memory
from langchain.memory import ConversationBufferMemory

# For identifying message roles
from langchain.schema import AIMessage, HumanMessage, SystemMessage

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

# Use LangChain Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

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
                        # Update the model name if needed
                        model="gemini-2.0-flash",
                        google_api_key=api_key,
                        temperature=0.7,
                        retry=retry_policy
                    )
                    # Clear memory whenever a new API key is set
                    st.session_state.memory.clear()
                    st.success("API Key configured successfully")
                except Exception as e:
                    st.error(f"Invalid API Key or authentication error: {e}")
            else:
                st.session_state.llm = None

    with st.container():
        st.markdown("<h3 style='margin-bottom: 10px;'>Conversation Controls</h3>", unsafe_allow_html=True)
        if st.button("Clear Conversation History"):
            st.session_state.memory.clear()
            st.success("Conversation history cleared!")

# -----------------------
# 4. Chat UI
# -----------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("<h3 style='margin-bottom: 20px;'>Conversation</h3>", unsafe_allow_html=True)
    
    # Display chat history from memory
    for msg in st.session_state.memory.chat_memory.messages:
        # Identify the role of each message (Human, AI, System)
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Tutor:** {msg.content}")
        elif isinstance(msg, SystemMessage):
            st.markdown(f"_System_: {msg.content}")

    # Chat input
    user_input = st.chat_input("Ask your data science question...")

# -----------------------
# 5. Handle Input & Response
# -----------------------
if user_input and st.session_state.llm:
    with st.spinner("Thinking..."):
        # 1) Add the user's message to memory as a HumanMessage
        st.session_state.memory.chat_memory.add_user_message(user_input)

        # 2) Build the final message list, starting with a system prompt
        messages = [{"role": "system", "content": "You are a helpful AI Data Science Tutor. Keep responses technical and concise."}]

        # Convert each message in memory to the 'role' format that ChatGoogleGenerativeAI expects
        for msg in st.session_state.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = "system"  # fallback

            messages.append({"role": role, "content": msg.content})

        # 3) Invoke Gemini LLM
        try:
            response = st.session_state.llm.invoke(messages)
            ai_response = response.content

            # 4) Add the AI's response back into the memory
            st.session_state.memory.chat_memory.add_ai_message(ai_response)

            # 5) Rerun to show the updated conversation
            st.rerun()

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

elif user_input:
    st.warning("Please enter your API Key first")
