"""
train_baseline.py — the centrally-trained baseline risk classifier.

Trains one logistic regression on ALL institutions' pooled data (i.e. the
"if we ignored privacy and centralised everything" comparison point for the
Federated Learning evaluation chart, Section 17/11).

Run:  python -m app.risk_model.train_baseline
"""
import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from app.data.oulad_prep import load_features, FEATURE_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "db_files", "risk_model_baseline.joblib")
RESULT_PATH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "db_files", "baseline_results.json")


def train_and_eval(seed=42):
    df = load_features()
    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)[:, 1]

    results = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, MODEL_PATH)
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("Centralized baseline results:", results)
    return results


if __name__ == "__main__":
    train_and_eval()
