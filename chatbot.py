import streamlit as st
import random
import torch as m  # Renaming torch to m
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-small"  # Choose a suitable conversational model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def text_to_speech(text):
    """Convert text to speech and save as an audio file."""
    tts = gTTS(text=text, lang='en')
    audio_file = "response.mp3"
    tts.save(audio_file)
    return audio_file

def get_response(user_input, chat_history_ids):
    """Get a response from the chatbot."""
    # Encode the new user input, add the eos_token, and return a tensor in PyTorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = m.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.numel() > 0 else new_user_input_ids

    # Generate a response from the model
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Get the response text
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Initialize chat history
chat_history_ids = m.tensor([]).reshape(0, 0)

# Streamlit UI setup
st.title("AI Chatbot")
st.write("Ask me anything about Artificial Intelligence and Computer Science!")

# User input
user_input = st.text_input("You: ")

if user_input:
    # Get response from the chatbot
    answer, chat_history_ids = get_response(user_input, chat_history_ids)
    
    # Display the chatbot's response
    st.text("Chatbot: " + answer)
    
    # Convert the response to speech
    audio_file = text_to_speech(answer)
    st.audio(audio_file, format='audio/mp3')

