import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.title("📰 Fake News Detection")

# Load dataset
data = pd.read_csv("news.csv")

# Split data
X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X, y)

# User input
user_input = st.text_area("Enter News Text")

if st.button("Check News"):
    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)

    st.subheader("Result:")
    st.write(prediction[0])