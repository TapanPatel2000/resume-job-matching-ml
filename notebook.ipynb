# Resume and Job Description Matching using Machine Learning

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------
# Load dataset
# -------------------------
data = pd.read_csv("data/sample_data.csv")  # Correct path to CSV inside data/ folder

print("Dataset shape:", data.shape)
print("\nLabel distribution:")
print(data['label'].value_counts())

# -------------------------
# Text preprocessing
# -------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

data['resume_clean'] = data['resume_text'].apply(preprocess_text)
data['job_clean'] = data['job_description_text'].apply(preprocess_text)
data['combined_text'] = data['resume_clean'] + " " + data['job_clean']

# -------------------------
# Train/test split
# -------------------------
X = data['combined_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------
# TF-IDF Vectorization
# -------------------------
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------------------------
# Model training
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -------------------------
# Model evaluation
# -------------------------
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# Confusion matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.show()

# -------------------------
# Conclusions
# -------------------------
print("""
Conclusions:

- This project demonstrates an end-to-end machine learning pipeline for text-based classification using NLP and classical ML.
- TF-IDF effectively transformed unstructured text into numerical features.
- Logistic Regression provided a strong and interpretable baseline.
- The focus was on understanding the full ML lifecycle rather than maximizing accuracy.

Future Improvements:

- Expand dataset size
- Use semantic embeddings (e.g., BERT)
- Experiment with additional models and evaluation metrics
""")
