import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the text file and preprocess the data
file_path = r'C:\Users\benslimane\PycharmProjects\pythonProjectspeach_recogni\.venv\Scripts\ww2.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence

# Define a function to interact with the chatbot
def chatbot(input_text):
    if input_text.lower() == "exit":
        return "Goodbye!"
    else:
        return get_most_relevant_sentence(input_text)

# Initialize the speech recognition recognizer
recognizer = sr.Recognizer()

# Streamlit UI
st.title("Speech-Enabled Chatbot")
st.write("You can ask questions by typing or speaking. Type 'exit' to end the conversation.")

# Get the user's input (speech or text)
input_type = st.radio("Select input type:", ["Text", "Speech"])

if input_type == "Text":
    # Text input
    question = st.text_input("You:")
    if st.button("Submit"):
        response = chatbot(question)
        st.write("Chatbot:", response)
else:
    # Speech input
    st.write("Press the button and speak into the microphone to recognize your speech.")

    def recognize_speech():
        try:
            with sr.Microphone() as source:
                st.text("Listening... Speak something")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                st.text("Recognizing...")

                text = recognizer.recognize_google(audio)
                st.write(f"You said: {text}")
                response = chatbot(text)
                st.write("Chatbot:", response)
        except sr.UnknownValueError:
            st.write("Google Web Speech API could not understand audio.")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Web Speech API; {e}")
        except Exception as e:
            st.write(f"An error occurred: {e}")

    if st.button("Start Recognition"):
        recognize
