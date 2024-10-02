import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import joblib

# Load the dataset
data = pd.read_csv('vaccination_all_tweets.csv')

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical features
data['user_name_enc'] = le.fit_transform(data['user_name'])
data['user_location_enc'] = le.fit_transform(data['user_location'])
data['user_description_enc'] = le.fit_transform(data['user_description'])
data['text_enc'] = le.fit_transform(data['text'])
data['hashtags_enc'] = le.fit_transform(data['hashtags'])

# Define features and target variable
X = data.drop('is_retweet', axis=1)
y = data['is_retweet']  

# Select features for training the model
features = X[['user_name_enc', 'user_location_enc', 'user_description_enc', 'text_enc', 'hashtags_enc', 'retweets', 'favorites']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict the outcomes for the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file
joblib.dump(model, 'decision_tree_model.pkl')
print("Model saved as decision_tree_model.pkl")

# Function for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Polarity score between -1 (negative) and 1 (positive)

    # Determine sentiment result based on polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to the 'text' column of your dataset
data['sentiment_result'] = data['text'].apply(analyze_sentiment)

# Optionally encode the sentiment results for further modeling
data['sentiment_encoded'] = data['sentiment_result'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

# Print the updated DataFrame with sentiment results
print(data[['text', 'sentiment_result', 'sentiment_encoded']].head())