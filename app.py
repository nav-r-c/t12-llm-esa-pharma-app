import streamlit as st
from src.rag import get_answer  # Assuming your get_answer function is in the src/rag.py file
from streamlit_local_storage import LocalStorage  # LocalStorage package to persist chat history

# Initialize local storage object
local_storage = LocalStorage()

# Helper function to save chat history
def save_to_local_storage(key, history):
    local_storage.setItem(key, history)

# Helper function to load chat history
def load_from_local_storage(key):
    return local_storage.getItem(key) or []

# Initialize histories from local storage
qa_history = load_from_local_storage("qa_history")

# Title
st.title("Pharma Knowledge Assistant")

# User query input and buttons in the same row
col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_area("Enter your query about pharmaceutical products:")
with col2:
    submit_button = st.button("Submit")
    clear_button = st.button("Clear History")

# Display chat history in the sidebar
def display_history(history):
    if history:
        st.sidebar.write("### Conversation History")
        for idx, chat in enumerate(history[::-1], start=1):
            st.sidebar.write(f"*You:* {chat['query']}")
            st.sidebar.write(f"*Assistant:* {chat['response']}")
            st.sidebar.markdown("---")
    else:
        st.sidebar.write("No conversation history yet. Start by asking a question!")

# Handle form submission
if submit_button and user_query.strip():
    with st.spinner("Generating response..."):
        response = get_answer(user_query)
        
    # Save the current question and response to chat history
    qa_history.append({"query": user_query, "response": response})
    save_to_local_storage("qa_history", qa_history)

    # Display the assistant's response below the form
    st.write("### Assistant's Response")
    st.write(response)

# Clear history functionality
if clear_button:
    qa_history = []
    save_to_local_storage("qa_history", qa_history)

# Display chat history in the sidebar
display_history(qa_history)

# Footer
st.sidebar.write("Team 12")
