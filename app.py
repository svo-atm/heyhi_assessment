#!/usr/bin/env python3
"""
Streamlit Web Interface for Educational RAG Chatbot
Deploy this app to make the chatbot accessible via web browser.
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Handle API key from Streamlit secrets or environment
try:
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    # Otherwise, the key should already be set in chatbot.py
except:
    pass

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from chatbot import create_parent_chain, vectorstore, chat_model, embeddings
    from langchain.storage import LocalFileStore
    from langchain.storage._lc_store import create_kv_docstore
    CHATBOT_AVAILABLE = True
except ImportError as e:
    CHATBOT_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="Educational RAG Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stChat > div {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
.assistant-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

def initialize_chatbot():
    """Initialize the chatbot components."""
    if "chatbot_initialized" not in st.session_state:
        try:
            store = {}
            fs = LocalFileStore("./store_location")
            parent_store = create_kv_docstore(fs)
            chain = create_parent_chain(vectorstore, store, parent_store)
            
            st.session_state.chatbot_initialized = True
            st.session_state.chain = chain
            st.session_state.store = store
            return True
        except Exception as e:
            st.session_state.chatbot_error = str(e)
            return False
    return True

def get_chatbot_response(question, session_id="streamlit_session"):
    """Get response from the chatbot."""
    try:
        if "chain" not in st.session_state:
            return "❌ Chatbot not initialized properly."
        
        response = st.session_state.chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
        return response
    except Exception as e:
        return f"❌ Error getting response: {str(e)}"

def main():
    """Main Streamlit app."""
    
    # Header
    st.title("🧬 Educational RAG Chatbot")
    st.markdown("*Learn about cells and chemistry of life with AI assistance*")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About This Chatbot")
        st.markdown("""
        This AI chatbot helps primary school students learn about:
        - **Cell Biology**: What are cells, their structures, and functions
        - **Scientific Discovery**: How cells were discovered and studied
        - **Microscopy**: Tools scientists use to observe cells
        - **Chemistry of Life**: Basic biological processes
        """)
        
        st.header("💡 Example Questions")
        example_questions = [
            "What are cells and why are they important?",
            "Who discovered cells?",
            "How do microscopes work?",
            "What is protoplasm?",
            "Why are cells called building blocks of life?"
        ]
        
        for i, question in enumerate(example_questions, 1):
            if st.button(f"💬 {question}", key=f"example_{i}"):
                st.session_state.current_question = question
    
    # Check if chatbot is available
    if not CHATBOT_AVAILABLE:
        st.error(f"❌ **Chatbot Components Not Available**")
        st.error(f"Import Error: {IMPORT_ERROR}")
        st.markdown("""
        **To fix this:**
        1. Ensure you're running from the correct directory
        2. Install dependencies: `pip install -r requirements.txt`
        3. Set up PostgreSQL with PGVector
        4. Configure your OpenAI API key
        """)
        return
    
    # Initialize chatbot
    if not initialize_chatbot():
        st.error("❌ **Failed to Initialize Chatbot**")
        if "chatbot_error" in st.session_state:
            st.error(f"Error: {st.session_state.chatbot_error}")
        st.markdown("""
        **Common issues:**
        - Database connection problems
        - Missing data files
        - API key configuration
        """)
        return
    
    st.success("✅ **Chatbot Ready!** Ask questions about cells and chemistry of life.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! I'm here to help you learn about cells and chemistry of life. What would you like to know? 🧬"
            }
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle example question selection
    if "current_question" in st.session_state:
        user_input = st.session_state.current_question
        del st.session_state.current_question
    else:
        # Chat input
        user_input = st.chat_input("Ask me about cells, microscopes, or biology...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chatbot_response(user_input)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! I'm here to help you learn about cells and chemistry of life. What would you like to know? 🧬"
            }
        ]
        st.rerun()

if __name__ == "__main__":
    main()
