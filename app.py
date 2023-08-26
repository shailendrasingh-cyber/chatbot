import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="./images/logob.png", width=100, height=60)
st.sidebar.image(my_logo)
def main():
    global counter
    st.title("Baba Python Chatbot")
    #st.sidebar.image(add_logo(logo_path="your/logo/path", width=50, height=60)) 

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. आपका संदेश")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("Welcome to Baba Python's innovative chat bot powered by advanced natural language processing (NLP) techniques and presented using the Streamlit web framework. We are excited to introduce you to a sophisticated solution that merges AI-driven conversations with a user-friendly interface, providing an interactive and seamless communication experience. ")

        st.subheader("Technological Brilliance:")

        st.write("""
        The project is divided into three  parts:
        1. NLP and Intent Recognition:  Our chat bot is powered by NLP techniques, enabling it to understand user intentions and respond contextually.
        2. Logistic Regression Algorithm: We've harnessed the power of Logistic Regression to categorize user inputs and formulate suitable responses.
        3. Streamlit Framework: The intuitive and visually appealing interface is built using the Streamlit web framework, ensuring seamless user interaction.
        """)

        st.subheader("Dataset:")

        st.write("""
       The project's dataset comprises meticulously annotated intents and entities, meticulously organized within a list structure.
       It encompasses the following key components:
       Intents: These encapsulate the underlying purpose of user inputs,
       featuring categories such as "greeting," "budget," and "about."
       Entities: Extracted from user inputs, entities encompass details that warrant specific attention, 
       embodying variations like "Hi," "How do I create a budget?" and "What is your purpose?"
       Text: The textual content input by users, which serves as the basis for the entire interaction.
        """)

        st.subheader("Made By ")

        st.write("The chatbot interface is built using Streamlit  and created by BabaPython  // visit :- babapython.pythonanywhere.com")

        st.subheader("Conclusion:")

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()