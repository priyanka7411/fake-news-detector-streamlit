import streamlit as st
import pickle
import numpy as np
import pandas as pd
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyttsx3

# Load the trained model and vectorizer with error handling
try:
    model = pickle.load(open('fake_news_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Function to predict the news class
def predict_news(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

# Function to generate word cloud
def generate_wordcloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Function to speak the article text
def speak_article(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load datasets for fake and real news articles (assuming CSV files)
fake_df = pd.read_csv('D:/Fake News Detection/News _dataset/Fake.csv')
true_df = pd.read_csv('D:/Fake News Detection/News _dataset/True.csv')

# Combine both datasets for prediction (fake and true news)
df = pd.concat([fake_df, true_df], ignore_index=True)

# Streamlit custom CSS for styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #2C3E50;
            font-size: 40px;
            font-family: 'Verdana', sans-serif;
        }
        .header {
            background-color: #34495E;
            padding: 20px;
            color: white;
            text-align: center;
            font-size: 30px;
        }
        .button {
            background-color: #E74C3C;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .container {
            display: flex;
            justify-content: space-between;
            padding: 20px;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Apply custom styles to your app
st.markdown('<div class="title">📰 Fake News Detection</div>', unsafe_allow_html=True)

# Choose the subject from the list
subject_options = ['News', 'politics', 'worldnews', 'politicsNews', 'Middle-east', 'left-news', 'Government News', 'US_News']
selected_subject = st.selectbox("Select a news subject:", subject_options)

# Filter news articles by the selected subject
filtered_df = df[df['subject'] == selected_subject]  # Assuming your dataset has a 'subject' column
shuffled_articles = random.sample(list(filtered_df['text']), len(filtered_df))  # Shuffle news articles

# Show a list of shuffled news articles
selected_article = st.selectbox("Select a news article:", shuffled_articles)

# Display full article in a text area for better reading
st.text_area("Full Article", selected_article, height=200)



# Predict when user selects the article
if st.button('Predict'):
    st.spinner('Predicting...')
    prediction = predict_news(selected_article)
    st.success('Done!')

    # Display result with emojis for more engagement
    if prediction == 'FAKE':
        st.markdown("<h2 style='color:red;'>❌ FAKE News</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:green;'>✅ REAL News</h2>", unsafe_allow_html=True)

# Generate a word cloud for the selected subject's news
st.subheader(f"Word Cloud for {selected_subject} Articles")
generate_wordcloud(filtered_df['text'].tolist())

# Add a loading spinner while performing predictions or visualizations
with st.spinner('Loading...'):
    # This can be placed around your main functionality, like fetching data or rendering predictions
    pass
