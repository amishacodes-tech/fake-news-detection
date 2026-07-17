import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Dono files load karein
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# 2. Labels lagayein
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# 3. Yahan merge ho raha hai (yahi wo line hai jo pehle miss thi)
df = pd.concat([fake_df, true_df], ignore_index=True)

# 4. X aur y define karein
X = df['text']
y = df['label']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# 7. Model train
pac = LogisticRegression()
pac.fit(tfidf_train, y_train)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, pac.predict(tfidf_vectorizer.transform(X_test))))

# 8. Save
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(pac, f)

print("🎉 Success! Naye tarike se model save ho gaya hai!")