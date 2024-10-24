import streamlit as st
from transformers import pipeline

# Load the Hugging Face pipeline
@st.cache_resource
def load_model():
    # Using GPT-Neo for text generation
    return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# Function to generate AI responses
def generate_response(prompt, model):
    response = model(prompt, max_length=300, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

# Streamlit UI
def main():
    st.title("AI Chatbot - Ask Anything About Artificial Intelligence")
    st.write("This chatbot uses a GPT-Neo model to answer questions about Artificial Intelligence.")
    
    # Text input for the user question
    user_input = st.text_input("Ask your AI-related question here:")
    
    if user_input:
        # Load the model
        model = load_model()
        
        # Generate the response
        with st.spinner('Thinking...'):
            ai_response = generate_response(user_input, model)
        
        # Display the response
        st.write(ai_response)

if __name__ == "__main__":
    main()
