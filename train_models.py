import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("Step 1: Loading Kaggle dataset...")
# Make sure you have downloaded Reviews.csv from Kaggle and placed it in this folder
df = pd.read_csv("Reviews.csv")

# Clean data
df = df.dropna(subset=['Text', 'Score'])

# Take a balanced sample of 20,000 reviews to train quickly without crashing
df = df.sample(n=20000, random_state=42)

print("Step 2: Converting text to numbers...")
# TF-IDF Vectorizer
X = df['Text']
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X_tfidf = vectorizer.fit_transform(X)

print("Step 3: Training the 'Boss' Rating Model (1 to 5 Stars)...")
# Target: Exact star rating
y_rating = df['Score']
rating_model = LogisticRegression(class_weight='balanced', max_iter=1000)
rating_model.fit(X_tfidf, y_rating)

print("Step 4: Training Buying Probability Model...")
# Target: 1 if Score >= 4 (Worth Buying), 0 otherwise
y_buy = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
buy_model = LogisticRegression(class_weight='balanced', max_iter=1000)
buy_model.fit(X_tfidf, y_buy)

print("Step 5: Saving the new models...")
# Save files for Streamlit
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(rating_model, 'rating_model.pkl')
joblib.dump(buy_model, 'buy_model.pkl')

print("✅ SUCCESS! Fresh models are saved. You can now run your Streamlit app.")