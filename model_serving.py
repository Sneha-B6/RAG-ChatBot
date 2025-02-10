from flask import Flask, request, jsonify
import pickle
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

# Define a root route (this will respond to GET requests at the root URL)
@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API! Use /predict for predictions."

@app.route('/predict', methods=['GET'])
def predict():
    try:
        review_text = request.args.get('review_text')  # Extract query parameter

        if not review_text:
            return jsonify({'error': 'No review_text provided'}), 400
        
        # Preprocess the review text the same way you did for training
        review_text = clean_text(review_text)
        
        # Transform the text using the TF-IDF vectorizer
        tfidf_input = vectorizer.transform([review_text])
        
        # Predict sentiment using the model
        sentiment = model.predict(tfidf_input)[0]
        
        return jsonify({'sentiment_prediction': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)}), 400



# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
