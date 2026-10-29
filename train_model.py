"""Model Training Module for Agricultural Question Topic Classification.

Trains a Multinomial Naive Bayes model using TF-IDF feature extraction inside
an end-to-end Scikit-learn Pipeline. Saves the trained pipeline to disk for inference.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import TextPreprocessor

DATASET_PATH = "agriculture_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_pipeline.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
TEST_DATA_PATH = os.path.join(MODEL_DIR, "heldout_test_data.csv")


def validate_dataset(df: pd.DataFrame) -> None:
    """Validates dataset structure, column schema, and data sufficiency."""
    required_cols = {"question", "topic"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if len(df) < 100:
        raise ValueError(f"Dataset too small ({len(df)} rows). Expected at least 100 samples.")

    unique_topics = df["topic"].nunique()
    if unique_topics < 12:
        raise ValueError(f"Found only {unique_topics} topic classes. Expected at least 12.")

    # Check nulls
    null_counts = df[["question", "topic"]].isnull().sum()
    if null_counts.any():
        print(f"Warning: Found missing values:\n{null_counts}. Dropping null rows.")
        df.dropna(subset=["question", "topic"], inplace=True)


def train():
    """Main training routine."""
    print("=" * 65)
    print("AGRICULTURE TOPIC CLASSIFIER - MULTINOMIAL NAIVE BAYES TRAINING")
    print("=" * 65)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file '{DATASET_PATH}' was not found. Please provide it first.")

    print(f"[1/5] Loading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    validate_dataset(df)

    # Check duplicates
    initial_len = len(df)
    df.drop_duplicates(subset=["question"], keep="first", inplace=True)
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} duplicate questions. Remaining: {len(df)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[2/5] Dataset validated: {len(df)} samples across {df['topic'].nunique()} classes.")
    topic_counts = df["topic"].value_counts()
    print("Class distribution summary:")
    for topic, count in topic_counts.items():
        print(f"  - {topic:<25}: {count} questions")

    # Train/Test Split (Stratified, reproducible seed)
    print("\n[3/5] Splitting dataset into training (80%) and testing (20%) sets...")
    X = df["question"]
    y = df["topic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save held-out test split for evaluate_model.py
    test_df = pd.DataFrame({"question": X_test, "topic": y_test})
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Held-out test dataset saved to '{TEST_DATA_PATH}' ({len(test_df)} samples).")

    # Define End-to-End Pipeline
    print("\n[4/5] Building Pipeline (Preprocessor -> TfidfVectorizer -> MultinomialNB)...")
    pipeline = Pipeline([
        ("preprocessor", TextPreprocessor(max_len=1000)),
        ("vectorizer", TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=1,
            sublinear_tf=True,
            norm="l2"
        )),
        ("classifier", MultinomialNB(alpha=0.5, fit_prior=True))
    ])

    print("Fitting model on training data...")
    pipeline.fit(X_train, y_train)

    # Evaluation on Train & Test
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\n[5/5] Training Complete!")
    print(f"  Training Accuracy : {train_acc * 100:.2f}%")
    print(f"  Testing Accuracy  : {test_acc * 100:.2f}%")

    # Save trained pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline artifact to '{MODEL_PATH}'.")

    # Save model metadata
    metadata = {
        "model_type": "MultinomialNB",
        "vectorizer": "TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)",
        "smoothing_alpha": 0.5,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_classes": int(df["topic"].nunique()),
        "classes": sorted(df["topic"].unique().tolist()),
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to '{METADATA_PATH}'.")

    print("\nQuick verification on sample questions:")
    samples = [
        "Why are my rice plants turning yellow with brown spots?",
        "What is the best subsidy scheme for buying a tractor?",
        "Current wholesale market rate of onions in Lasalgaon mandi"
    ]
    for s in samples:
        pred = pipeline.predict([s])[0]
        probs = pipeline.predict_proba([s])[0]
        conf = np.max(probs)
        print(f"  Q: '{s}'")
        print(f"  --> Predicted: {pred} (Confidence: {conf*100:.1f}%)\n")

    return pipeline


if __name__ == "__main__":
    train()
