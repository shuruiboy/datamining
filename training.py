import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('python-tweets.csv')

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

df['Tweet'] = df['Tweet'].apply(preprocess)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Sentiment'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5)

# Fit the vectorizer on the training data and transform both the training and testing data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Accuracy score
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
probas = clf.predict_proba(X_test_tfidf)

#Save the model
joblib.dump(clf, 'model.joblib')
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Create a scatter plot
plt.figure(figsize=(8,6))
c = y_test.map({'negative': -1, 'neutral': 0, 'positive': 1}).fillna(0).astype(int)
plt.scatter(np.arange(len(probas)), np.max(probas, axis=1), c=c, cmap='coolwarm')
plt.xlabel('Test Instances')
plt.ylabel('Confidence')
plt.title(f'Classifier Confidence on Test Data (Accuracy = {accuracy:.2f})')
plt.colorbar(ticks=[0, 1, 2], label='Sentiment')
plt.show()
