import os

os.environ["MPLCONFIGDIR"] = "/tmp"  # Prevent matplotlib config errors
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Title and UI
st.set_page_config(page_title="DeepSeek-R1 Chatbot", page_icon="🤖")
st.title("🧠 MentorMind AI - Your Intelligent Academic & Career Co-Pilot")
st.caption("Running entirely on CPU using Hugging Face Transformers")


# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
    return tokenizer, model


tokenizer, model = load_model()

st.chat_message("ai").write("What do you want to learn today?")
user_input = st.chat_input("I want to learn...")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display assistant response in chat message container
        st.chat_message("ai").markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
