from flask import Flask, render_template, request
import joblib
import pandas as pd
from textblob import TextBlob

app = Flask(__name__)

# Load the trained Decision Tree model (if needed for retweet prediction)
model = joblib.load('decision_tree_model.pkl')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data from the user input
        user_name = request.form['user_name']
        user_location = request.form['user_location']
        user_description = request.form['user_description']
        text = request.form['text']
        hashtags = request.form['hashtags']
        retweets = int(request.form['retweets'])
        favorites = int(request.form['favorites'])

        # Prepare data for retweet prediction (if needed, otherwise remove)
        input_data = pd.DataFrame([[user_name, user_location, user_description, text, hashtags, retweets, favorites]],
                                  columns=['user_name', 'user_location', 'user_description', 'text', 'hashtags', 'retweets', 'favorites'])
        
        # Retweet prediction using model (optional, can be removed)
        prediction = model.predict(input_data)[0]
        retweet_result = 'Retweet' if prediction == 1 else 'Not Retweet'
        
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

        return render_template('index.html', retweet_result=retweet_result, sentiment_result=sentiment_result)

if __name__ == '__main__':
    app.run(debug=True)
