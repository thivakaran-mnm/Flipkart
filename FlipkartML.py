import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import streamlit as st

# Function to load and preprocess data
def load_data():
    df = pd.read_csv("E:\INNOMATICS\data.csv")
    # Preprocessing steps can be added here
    return df

# Function to train model
def train_model(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    
    # Check if the column 'Review_Text' exists in the dataframe
    if 'Review_Text' in df.columns:
        X = tfidf_vectorizer.fit_transform(df['Review_Text'])
    else:
        st.error("Column 'Review_Text' not found in the dataset.")
        return None, None, None
    
    # Check if the column 'Rating' exists in the dataframe
    if 'Rating' in df.columns:
        y = df['Rating'] > 3  # Consider ratings greater than 3 as positive sentiment
    else:
        st.error("Column 'Rating' not found in the dataset.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    return model, tfidf_vectorizer, f1

# Function to predict sentiment
def predict_sentiment(review_text, model, vectorizer):
    review_vectorized = vectorizer.transform([review_text])
    prediction = model.predict(review_vectorized)
    return prediction[0]

# Load data
df = load_data()

# If dataframe is loaded successfully, train model
if df is not None:
    # Train model
    model, tfidf_vectorizer, f1 = train_model(df)

    # Streamlit app
    st.set_page_config(page_title="Flipkart Review Sentiment Analysis", page_icon=":smiley:")

    def main():
        st.title('Flipkart Review Sentiment Analysis')

        menu = ['Home', 'About', 'Predict']
        choice = st.sidebar.selectbox('Menu', menu)

        if choice == 'Home':
            st.write('Welcome to Flipkart Review Sentiment Analysis App!')

        elif choice == 'About':
            st.header('About')
            st.write('This app performs sentiment analysis on Flipkart reviews.')
            st.write('It uses a logistic regression model trained on the dataset of Flipkart reviews to predict whether a review expresses positive or negative sentiment.')

        elif choice == 'Predict':
            st.header('Predict Sentiment')
            review_text = st.text_area('Enter your review:', '')
            if st.button('Predict'):
                # Predict sentiment
                sentiment = predict_sentiment(review_text, model, tfidf_vectorizer)
                result_text = "Positive" if sentiment else "Negative"
                # Display result
                st.write(f'The sentiment of your review is: {result_text}')

    # Display F1 score if model is trained successfully
    if model is not None:
        st.sidebar.markdown(f'**F1 Score:** {f1:.2f}')
    else:
        st.sidebar.markdown('Model training failed. Please check the dataset.')

    if __name__ == "__main__":
        main()
