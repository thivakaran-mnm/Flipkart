import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import joblib

# Load dataset
# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\hp\Downloads\reviews_data_dump\reviews_badminton\data.csv")
    return df


# Preprocessing function
def preprocess_reviews(df, review_column):
    # Remove stopwords, punctuation, and special characters
    df[review_column] = df[review_column].str.lower().str.replace('[^\w\s]', '')
    # Fill missing values with empty strings
    df[review_column] = df[review_column].fillna('')
    # Perform lemmatization or stemming (not implemented here)
    return df

# Feature extraction function
def extract_features(df, review_column):
    # Use TF-IDF vectorizer for feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[review_column])
    y = df['Ratings']  # Change to the appropriate column name
    return X, y

# Model training function
def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    # Evaluate the model
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return clf, f1

# Streamlit app
def main():
    st.title("Badminton Product Reviews Sentiment Analysis")
    
    # Load dataset
    df = load_data()

    # Display reviews
    st.header("Customer Reviews")
    st.write(df)

    # Select review column
    review_column = st.selectbox("Select the review column:", df.columns)

    # Preprocess and extract features
    df = preprocess_reviews(df, review_column)
    X, y = extract_features(df, review_column)

    # Train model
    st.text("Training model...")
    clf, f1 = train_model(X, y)
    st.text(f"Model trained successfully. F1 Score: {f1:.2f}")

    # Save model
    joblib.dump(clf, 'sentiment_model.joblib')
    st.text("Model saved successfully.")

    # Load saved model
    clf = joblib.load('sentiment_model.joblib')

    # Sentiment analysis
    st.header("Sentiment Analysis")
    review_text = st.text_input("Enter a review:")
    if review_text:
        review_text = preprocess_reviews(pd.DataFrame({review_column: [review_text]}), review_column)[review_column].iloc[0]
        X_test = vectorizer.transform([review_text])
        prediction = clf.predict(X_test)
        st.write(f"Predicted Rating: {prediction[0]}")

if __name__ == "__main__":
    main()
