import os

os.environ["MPLCONFIGDIR"] = "/tmp"  # Prevent matplotlib config errors
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import requests
import streamlit as st

# Title and UI
st.set_page_config(page_title="Gemini-2.0-flash Chatbot", page_icon="🤖")
st.title("🧠 MentorMind AI - Your Intelligent Academic & Career Co-Pilot")
st.caption("gemini-2.0-flash")


st.chat_message("ai").write("What do you want to learn today?")
user_input = st.chat_input("I want to learn...")


url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
api_key = os.environ["GOOGLE_GEMINI_API_KEY"]
headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}


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

    promot = f"""
      The user will enter a topic they want to learn.
      Your task is to create a simple, step-by-step learning roadmap for that topic.
      Keep it short, clear, and beginner-friendly — no technical jargon or deep explanations.
      Focus only on what's essential to get started and make steady progress.
      Make sure the steps are easy to follow and realistic for someone new to the topic.

      Always start with cli commands and code templates, and explain briefly.

      {user_input}
    """

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        payload = {"contents": [{"parts": [{"text": user_input}]}]}

        response = requests.post(url, headers=headers, json=payload)

        status_code = response.status_code

        if status_code == 200:
            data = response.json()

            text = data["candidates"][0]["content"]["parts"][0]["text"]

            # Display assistant response in chat message container
            st.chat_message("ai").markdown(text)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": text})
        else:
            st.chat_message("ai").markdown(
                f"Something went wrong! Error: {status_code}"
            )
            print(response.json())
