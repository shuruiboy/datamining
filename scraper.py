import pandas as pd
from tqdm.notebook import tqdm
import snscrape.modules.twitter as sntwitter
from transformers import pipeline

# Load the sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

query = ""
tweets = []
limits = 5000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limits:
        break
    else: 
        tweet_text = tweet.content
        tweet_sentiment = sentiment_analysis(tweet_text)[0]['label']
        if tweet_sentiment == 'POSITIVE':
            tweets.append([tweet.date, tweet.user.username, tweet.content, 1])
        elif tweet_sentiment == 'NEGATIVE':
            tweets.append([tweet.date, tweet.user.username, tweet.content, -1])
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.content, 0])

df = pd.DataFrame(tweets, columns= ['Date','User','Tweet', 'Sentiment'])

print(df)

df.to_csv("python-tweets.csv", index=False)
