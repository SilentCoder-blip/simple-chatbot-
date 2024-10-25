import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os
from IPython.display import Audio, display

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-small"  # Choose a suitable conversational model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def text_to_speech(text):
    """Convert text to speech and play it."""
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    display(Audio("response.mp3", autoplay=True))

def get_response(user_input, chat_history_ids):
    """Get a response from the chatbot."""
    # Encode the new user input, add the eos_token, and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.numel() > 0 else new_user_input_ids

    # Generate a response from the model
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Get the response text
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Initialize chat history
chat_history_ids = torch.tensor([]).reshape(0, 0)

# Example questions for the chatbot
example_questions = [
    "What is artificial intelligence?",
    "Can you explain neural networks?",
    "What are the types of machine learning algorithms?",
    "Describe the Turing test.",
    "What are the properties of artificial intelligence?"
]

# AI definitions and properties for enhancing responses
ai_definitions = {
    "What is artificial intelligence?": "Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence. This includes problem-solving, understanding natural language, recognizing patterns, and making decisions.",
    "What are the properties of artificial intelligence?": "Key properties of AI include adaptability, learning from data, reasoning, problem-solving abilities, and the ability to understand natural language. AI systems can also exhibit traits like perception, social intelligence, and autonomy."
}

print("Welcome to the AI Chatbot! Type 'exit' to end the chat.")
print("\nExample Questions:")
for question in example_questions:
    print("- " + question)

while True:
    # User input
    user_input = input("\nYou: ")
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Check if the input is a predefined question for a specific response
    if user_input in ai_definitions:
        answer = ai_definitions[user_input]
    else:
        # Get response from the chatbot for non-predefined questions
        answer, chat_history_ids = get_response(user_input, chat_history_ids)
    
    print("Chatbot: " + answer)
    
    # Convert the response to speech
    text_to_speech(answer)
