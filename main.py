from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

import streamlit as st
import os
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import tempfile

load_dotenv()

# Initialize model
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Provide responses to user queries."),
        ("user", "Question: {question}")
    ]
)

st.title("Hi! I'm Shresth. Ask your queries")

# Session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = Ollama(model="llama3")
output_parser = StrOutputParser()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Correct way to use memory in LangChain
chain = LLMChain(prompt=prompt, llm=llm, memory=memory, output_parser=output_parser)

# Function to convert text to speech
def speak(text):
    """Convert text to speech and play it."""
    if not text:
        st.warning("No text to convert to speech!")
        return
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        tts.save(temp_file.name)
        st.audio(temp_file.name, format="audio/mp3")

# User Input
input_text = st.text_input("Type your question")
submit_button = st.button("Ask")

if submit_button and input_text:
    with st.spinner("Thinking..."):
        response_text = chain.invoke({"question": input_text})
        
        # ✅ Ensure response is a string (fix TypeError)
        if isinstance(response_text, dict):  
            response_text = response_text.get("text", "")  # Extract "text" from dictionary

        st.session_state.chat_history.append({"user": input_text, "bot": response_text})
        st.write(response_text)
        speak(response_text)

# Display Chat History
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**Shresth:** {chat['bot']}")

# Speech-to-Text
recognizer = sr.Recognizer()

# ✅ Check if microphone is available
try:
    mic = sr.Microphone()
    st.write("🎤 Microphone detected.")
except OSError:
    st.write("❌ No microphone found! Please check your settings.")

if st.button("Use Voice Input"):
    with mic as source:
        st.write("🎤 Listening... Speak now")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for background noise
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # Stops after 5 sec
            st.write("🎙 Processing voice input...")
            
            input_text = recognizer.recognize_google(audio)
            st.text(f"Recognized: {input_text}")

            # Process the recognized speech
            with st.spinner("Thinking..."):
                response_text = chain.invoke({"question": input_text})

                # ✅ Ensure response is a string
                if isinstance(response_text, dict):
                    response_text = response_text.get("text", "")

                st.session_state.chat_history.append({"user": input_text, "bot": response_text})
                st.write(response_text)
                speak(response_text)

        except sr.UnknownValueError:
            st.write("❌ Could not understand. Please try again.")
        except sr.WaitTimeoutError:
            st.write("⏳ No speech detected. Try speaking louder or closer to the mic.")
        except Exception as e:
            st.write(f"⚠ Error: {str(e)}")
