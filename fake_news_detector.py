#!/usr/bin/env python3
"""
Fake News Detection System
--------------------------
Now saves training metadata (feature column, ngram range) for correct evaluation.
"""

import os
import sys
import argparse
import logging
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATA_DIR = "dataset"
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

METADATA_FILE = os.path.join(MODEL_DIR, "training_metadata.json")


def save_metadata(feature_column, ngram_range):
    """Save training configuration to a JSON file."""
    metadata = {
        "feature_column": feature_column,
        "ngram_range": list(ngram_range)  # convert tuple to list for JSON
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
    logger.info(f"Training metadata saved to {METADATA_FILE}")


def load_metadata():
    """Load training configuration from JSON file."""
    if not os.path.exists(METADATA_FILE):
        logger.error(f"Metadata file {METADATA_FILE} not found. Did you train the model?")
        sys.exit(1)
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    return metadata["feature_column"], tuple(metadata["ngram_range"])


def load_data(feature_column='text'):
    """Load datasets and return the selected feature column."""
    logger.info(f"Loading datasets (using column: '{feature_column}')...")
    try:
        fake = pd.read_csv(FAKE_CSV)
        true = pd.read_csv(TRUE_CSV)
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        sys.exit(1)

    fake["label"] = 0
    true["label"] = 1
    data = pd.concat([fake, true], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    if feature_column not in data.columns:
        logger.error(f"Column '{feature_column}' not found. Available: {list(data.columns)}")
        sys.exit(1)

    X = data[feature_column]
    y = data["label"]
    logger.info(f"Total samples: {len(data)} (Fake: {len(fake)}, True: {len(true)})")
    return X, y


def preprocess_text(X, y, max_df=0.7, max_features=50000, ngram_range=(1,2), save_vectorizer=True):
    """Convert text to TF‑IDF features."""
    logger.info(f"Applying TF‑IDF vectorization (ngram_range={ngram_range})...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range
    )
    X_tfidf = vectorizer.fit_transform(X)
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    if save_vectorizer:
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    return X_tfidf, vectorizer


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def train_logistic_regression(X_train, y_train):
    logger.info("Training Logistic Regression with GridSearchCV...")
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }
    lr = LogisticRegression(random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    logger.info(f"Best LR params: {grid.best_params_}, best CV score: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_naive_bayes(X_train, y_train):
    logger.info("Training Multinomial Naive Bayes with GridSearchCV...")
    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    nb = MultinomialNB()
    grid = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    logger.info(f"Best NB params: {grid.best_params_}, best CV score: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"{model_name} Test Accuracy: {acc:.4f}")
    print(f"\n{model_name} Classification Report:\n", classification_report(y_test, y_pred))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"{model_name} ROC‑AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake','True'], yticklabels=['Fake','True'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name.replace(' ', '_')}_cm.png"))
    plt.show()


def save_model(model, filename):
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        logger.error(f"Model file {path} not found.")
        sys.exit(1)
    return joblib.load(path)


def predict_news(text, model, vectorizer):
    X_vec = vectorizer.transform([text])
    pred = model.predict(X_vec)[0]
    proba = model.predict_proba(X_vec)[0] if hasattr(model, "predict_proba") else None
    label = "Real" if pred == 1 else "Fake"
    if proba is not None:
        confidence = proba[pred]
        logger.info(f"Prediction: {label} (confidence: {confidence:.4f})")
    else:
        logger.info(f"Prediction: {label}")
    return label


def main():
    parser = argparse.ArgumentParser(description="Fake News Detection System")
    parser.add_argument("--train", action="store_true", help="Train models from scratch")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing models on test set")
    parser.add_argument("--predict", type=str, help="Predict a single news headline or text")
    parser.add_argument("--feature-column", type=str, choices=['text', 'title'], default='text',
                        help="Column to use as input: 'text' (full article) or 'title' (headline)")
    parser.add_argument("--ngram-range", type=int, nargs=2, default=[1,2],
                        help="N‑gram range, e.g. --ngram-range 1 3 for trigrams")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # --- PREDICTION MODE ---
    if args.predict:
        logger.info("Loading trained model and vectorizer...")
        model = load_model("best_model.pkl")
        vectorizer = load_model("tfidf_vectorizer.pkl")
        predict_news(args.predict, model, vectorizer)
        return

    # --- TRAINING MODE ---
    if args.train:
        # Load data using chosen feature column
        X_text, y = load_data(feature_column=args.feature_column)

        # Preprocess with specified n‑gram range
        X_tfidf, vectorizer = preprocess_text(
            X_text, y,
            max_df=0.7,
            max_features=50000,
            ngram_range=tuple(args.ngram_range),
            save_vectorizer=True
        )

        # Split and train
        X_train, X_test, y_train, y_test = split_data(X_tfidf, y, test_size=0.2)

        lr_model = train_logistic_regression(X_train, y_train)
        nb_model = train_naive_bayes(X_train, y_train)

        evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        evaluate_model(nb_model, X_test, y_test, "Naive Bayes")

        # Determine best model
        lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
        nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

        if lr_acc >= nb_acc:
            best_model = lr_model
            best_name = "LogisticRegression"
        else:
            best_model = nb_model
            best_name = "NaiveBayes"

        logger.info(f"Best model: {best_name} with test accuracy {max(lr_acc, nb_acc):.4f}")

        # Save models
        save_model(best_model, "best_model.pkl")
        save_model(lr_model, "logistic_regression.pkl")
        save_model(nb_model, "naive_bayes.pkl")

        # Save metadata for future evaluation
        save_metadata(args.feature_column, args.ngram_range)

        # Also save the test data indices if you want to reuse the exact split? Not needed if we reload with same random_state.
        # But we can store the test data texts for later inspection? Optional.

        logger.info("Training completed. You can now run --evaluate (it will use the saved metadata).")

    # --- EVALUATION MODE ---
    if args.evaluate:
        # Load metadata to know which feature column and ngram range were used
        feature_column, ngram_range = load_metadata()
        logger.info(f"Using metadata: feature_column='{feature_column}', ngram_range={ngram_range}")

        # Load data with the correct column
        X_text, y = load_data(feature_column=feature_column)

        # Preprocess with the correct ngram range
        # Note: We must use the SAME vectorizer that was saved during training.
        # Instead of creating a new vectorizer, we load the saved one.
        vectorizer = load_model("tfidf_vectorizer.pkl")
        # But we need to ensure the vectorizer's ngram_range matches; it should, because we saved it.
        X_tfidf = vectorizer.transform(X_text)  # transform, not fit_transform!

        # Recreate the exact same train/test split (random_state=42 ensures reproducibility)
        X_train, X_test, y_train, y_test = split_data(X_tfidf, y, test_size=0.2)

        # Load saved models
        logger.info("Loading pre‑trained models for evaluation...")
        try:
            lr_model = load_model("logistic_regression.pkl")
            nb_model = load_model("naive_bayes.pkl")
        except FileNotFoundError:
            logger.error("Models not found. Please run with --train first.")
            sys.exit(1)

        # Evaluate on the test set
        evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        evaluate_model(nb_model, X_test, y_test, "Naive Bayes")


if __name__ == "__main__":
    main()