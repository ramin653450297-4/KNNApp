from flask import Flask, render_template, request
import joblib
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.pkl')
le = LabelEncoder()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_name = request.form['user_name']
        user_location = request.form['user_location']
        user_description = request.form['user_description']
        text = request.form['text']
        hashtags = request.form['hashtags']
        retweets = int(request.form['retweets'])
        favorites = int(request.form['favorites'])

        # Encode categorical features
        user_name_enc = le.fit_transform([user_name])[0]  # Encode the user_name
        user_location_enc = le.fit_transform([user_location])[0]  # Encode the user_location
        user_description_enc = le.fit_transform([user_description])[0]  # Encode the user_description
        text_enc = le.fit_transform([text])[0]  # Encode the text
        hashtags_enc = le.fit_transform([hashtags])[0]  # Encode the hashtags

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[user_name_enc, user_location_enc, user_description_enc, text_enc, hashtags_enc, retweets, favorites]],
                                  columns=['user_name_enc', 'user_location_enc', 'user_description_enc', 'text_enc', 'hashtags_enc', 'retweets', 'favorites'])

        # Predict retweet classification
        retweet_prediction = model.predict(input_data)[0]  # Make prediction using the model

        # Sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # Polarity score between -1 (negative) and 1 (positive)

        # Determine sentiment result
        if sentiment > 0:
            sentiment_result = 'Positive'
        elif sentiment < 0:
            sentiment_result = 'Negative'
        else:
            sentiment_result = 'Neutral'

        # Return both predictions to the template
        return render_template('index.html', sentiment_result=sentiment_result, retweet_prediction=retweet_prediction)

if __name__ == '__main__':
    app.run(debug=True)
