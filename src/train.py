from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

import pickle

from src.preprocess import load_txt_dataset, preprocess_data

def train_model():
    # Load and preprocess data
    df = load_txt_dataset("data/train_data.txt")
    df = preprocess_data(df)

    # Prepare features and labels
    X = df['clean_description']
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genre'])

    # Save label binarizer
    with open("models/genre_binarizer.pkl", "wb") as f:
        pickle.dump(mlb, f)

    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("🛠 Training model...")
    pipeline.fit(X_train, y_train)

    # Save model
    with open("models/genre_classifier.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("✅ Model trained and saved.")
    return pipeline, X_test, y_test, mlb
