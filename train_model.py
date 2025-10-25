import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# 1. Load dataset
df = pd.read_csv("reviews.csv", quotechar='"')

# 2. Clean text
df['review_clean'] = df['review'].str.lower().str.translate(str.maketrans('', '', string.punctuation))
df['review_clean'] = df['review_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in ENGLISH_STOP_WORDS]))

# 3. Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review_clean'])
y = df['sentiment']

# 4. Split data & train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)


# 6. Save model and vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")

