"""
Simple evaluation script using scikit-learn.
- Loads the iris dataset
- Trains a RandomForestClassifier
- Prints accuracy and classification report
- Saves the trained model to ./models/latest.joblib
"""
from pathlib import Path
import argparse
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main(output_path: str, test_size: float, random_state: int):
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds, target_names=data.target_names))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"Saved model to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & evaluate a simple sklearn model.")
    parser.add_argument("--output", "-o", default="models/latest.joblib", help="Path to save trained model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args.output, args.test_size, args.random_state)