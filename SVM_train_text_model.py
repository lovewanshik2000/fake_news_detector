# 🚀 Upgraded Training Script (High Accuracy TF-IDF + SVM)
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODELS_DIR = "models"
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_model_svm_v4.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer_svm_v4.joblib")

DATA_PATH = "data/news.csv"  # your dataset path

def main():

    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Load & Clean Data ---
    df = pd.read_csv(DATA_PATH, sep=",", engine="python", on_bad_lines="skip")

    # Remove hidden tabs in text
    df["text"] = df["text"].astype(str).str.replace("\t", " ", regex=False)

    # Normalize labels
    df["label"] = df["label"].astype(str).str.strip().str.upper()

    # Drop empty rows
    df = df[df["text"].str.strip() != ""]
    df = df[df["label"].str.strip() != ""]

    X = df["text"]
    y = df["label"]

    print("\nClass distribution:")
    print(y.value_counts())

    # Avoid stratify errors
    if y.value_counts().min() >= 2:
        strat = y
    else:
        strat = None
        print("⚠ Not using stratify (class too small).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    # --- Stronger Model: Character-level TF-IDF + SVC WITH PROBABILITY ---
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=20000
        )),
        ("svm", SVC(kernel="linear", probability=True))
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = pipeline.predict(X_test)
    print("\n ################ Classification Report: ################")
    print(classification_report(y_test, y_pred))

    # --- Save ---
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)
    joblib.dump(pipeline.named_steps["svm"], TEXT_MODEL_PATH)

    print("\nSaved vectorizer to:", VECTORIZER_PATH)
    print("Saved model to:", TEXT_MODEL_PATH)

if __name__ == "__main__":
    main()