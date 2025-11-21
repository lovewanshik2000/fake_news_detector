import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Paths (match your Flask app)
MODELS_DIR = "models"
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_model_v3.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer_v3.joblib")

DATA_PATH = "data/news.csv"  # adjust if needed

def main():
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # Basic sanity check
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns.")

    X_text = df["text"].astype(str)
    y = df["label"].astype(str)

    # 2. Create and fit the vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=1,                # keep all terms (tiny dataset)
        max_features=5000        # cap features (optional)
    )
    X_vec = vectorizer.fit_transform(X_text)

    # 3. Train a simple classifier
    model = LogisticRegression(max_iter=2000)
    model.fit(X_vec, y)

    # 4. Save vectorizer & model
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, TEXT_MODEL_PATH)

    print("########### Training complete. ###########")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")
    print(f"Saved text model to: {TEXT_MODEL_PATH}")
    print("Classes:", model.classes_)

if __name__ == "__main__":
    main()

