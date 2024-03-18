# Flipkart Product Reviews Sentiment Analysis

This is a Streamlit web application for analyzing customer sentiment in product reviews scraped from Flipkart. The application allows users to view the dataset, train a sentiment analysis model, and perform real-time sentiment analysis on user-provided reviews.

## Installation

1. Clone the repository:
2. Navigate to the project directory:
3. Install the required dependencies
4. pip install -r requirements.txt

## Usage

1. Run the Streamlit app: streamlit run flipkart.py
2. Open your web browser and go to http://localhost:8501 to access the application.

You can view the dataset, train a sentiment analysis model, and perform real-time sentiment analysis on user-provided reviews using the web interface.

## File Structure

1. flipkart.py: Main Python script containing the Streamlit web application.
2. data.csv: Dataset containing Flipkart product reviews.
3. requirements.txt: List of Python dependencies required for the project.

## Features

1. Load and display the dataset.
2. Preprocess the review text and extract features for model training.
3. Train a sentiment analysis model using the Multinomial Naive Bayes classifier.
4. Save the trained model for later use.
5. Perform real-time sentiment analysis on user-provided reviews.
