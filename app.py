from flask import Flask, request, render_template
import numpy as np
import re
import pickle
import nltk
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load saved models and vectorizers
lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# Define text preprocessing function
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# Define prediction function
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]

    # Get the probability of the predicted class
    probability = np.max(lg.predict_proba(input_vectorized))

    return predicted_emotion, probability

# Define Flask route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form["user_input"]
    predicted_emotion, probability = predict_emotion(user_input)
    return render_template("index.html", prediction=predicted_emotion, probability=probability)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
