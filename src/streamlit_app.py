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

if user_input:
    st.chat_message("user").write(user_input)

    with st.spinner("Thinking..."):
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.chat_message("ai").write(response)
