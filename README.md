# Fake News Detection using Streamlit

## Overview
This project aims to detect fake news using machine learning techniques. It utilizes a machine learning model that classifies news articles as real or fake based on their content. The application is developed using Streamlit for the frontend, providing an interactive user interface where users can input news articles and get predictions.

## Features
- **Interactive UI:** A simple and user-friendly interface built using Streamlit to input and classify news articles.
- **Fake News Classification:** Uses a machine learning model (PassiveAggressiveClassifier) trained on text data to predict whether a given news article is fake or real.
- **Real-time Prediction:** Classifies news in real-time as the user inputs the text.
  
## Installation

### Prerequisites
Ensure you have Python 3.6 or later installed on your machine.

### Step 1: Clone the repository

git clone https://github.com/priyanka7411/fake-news-detector-streamlit.git
cd fake-news-detector-streamlit

### Step 2: Install required dependencies

You can install the necessary libraries using pip:
pip install -r requirements.txt

### Step 3: Run the application

To start the application, use the following command:
streamlit run app.py

### How It Works
Model Training: A PassiveAggressiveClassifier model is trained on a dataset of news articles labeled as fake or real. The model is serialized and saved as model.pkl.
Streamlit App: The app takes user input (news article) and preprocesses the text, which is then passed to the trained model to predict if the news is fake or real.
Prediction Display: The result is displayed on the web interface in real-time.

### Requirements
Python 3.x
Streamlit
Scikit-learn
Pandas
NumPy
Joblib
All dependencies are listed in requirements.txt.

### Contributing
If you would like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Contributions are always welcome!

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
Streamlit for the web app framework.
Scikit-learn for machine learning algorithms.
The Fake News Dataset for training the model.
