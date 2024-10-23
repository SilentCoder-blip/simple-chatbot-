import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to get AI response from OpenAI API
def get_ai_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can switch to 'gpt-3.5-turbo' or 'gpt-4'
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app layout
st.title("AI Question Answering Chatbot")
st.write("Ask me anything about Artificial Intelligence!")

# Text input for user's question
user_input = st.text_input("Your Question:")

# When the user submits a question
if user_input:
    with st.spinner("Thinking..."):
        ai_response = get_ai_response(user_input)
    st.write("### Answer:")
    st.write(ai_response)

# Information section about the bot
st.sidebar.title("About this chatbot")
st.sidebar.info(
    """
    This chatbot is powered by OpenAI's GPT-3 model and can answer questions related to Artificial Intelligence. 
    Ask me about AI concepts, algorithms, or applications.
    """
)
