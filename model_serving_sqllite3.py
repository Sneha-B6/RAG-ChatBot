from flask import Flask, request, jsonify
import pickle
import pandas as pd
import string
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model_filename = 'sentiment_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_filename, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function (same as used during training)
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    return text

# Define a root route
@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API! Use /predict for individual reviews or /upload for datasets."

# Route to handle predictions for a single review
@app.route('/predict', methods=['GET'])
def predict():
    try:
        review_text = request.args.get('review_text')  # Extract query parameter

        if not review_text:
            return jsonify({'error': 'No review_text provided'}), 400
        
        # Preprocess the review text
        review_text_cleaned = clean_text(review_text)
        
        # Transform the text using the TF-IDF vectorizer
        tfidf_input = vectorizer.transform([review_text_cleaned])
        
        # Predict sentiment using the model
        sentiment = model.predict(tfidf_input)[0]
        
        return jsonify({'review_text': review_text, 'sentiment_prediction': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route to handle predictions for a dataset
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if a file is included in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        
        # Load the dataset
        dataset = pd.read_csv(file)
        
        if 'review' not in dataset.columns:
            return jsonify({'error': 'Dataset must contain a "review" column'}), 400
        
        # Preprocess and predict sentiments for all reviews
        dataset['cleaned_review'] = dataset['review'].apply(clean_text)
        tfidf_input = vectorizer.transform(dataset['cleaned_review'])
        dataset['sentiment_prediction'] = model.predict(tfidf_input)
        
        # Return predictions as JSON
        results = dataset[['review', 'sentiment_prediction']].to_dict(orient='records')
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
