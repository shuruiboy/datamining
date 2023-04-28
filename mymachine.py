import pandas as pd
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import snscrape.modules.twitter as sntwitter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the saved model and tokenizer
clf = joblib.load('model.joblib')
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Preprocess the text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove stopwords and lemmatize
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

query = "elon"
tweets = []
limits = 50

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limits:
        break
    else: 
        tweet_text = tweet.content
        tweet_text = preprocess(tweet_text)
        tweet_text_tfidf = tfidf.transform([tweet_text])
        tweet_sentiment = clf.predict(tweet_text_tfidf)[0]
        if tweet_sentiment == 1:
            tweets.append([tweet.date, tweet.user.username, tweet.content, 'positive'])
        elif tweet_sentiment == -1:
            tweets.append([tweet.date, tweet.user.username, tweet.content, 'negative'])
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.content, 'neutral'])

df = pd.DataFrame(tweets, columns= ['Date','User','Tweet', 'Sentiment'])

print(df)

df.to_csv("mython-tweets.csv", index=False)
