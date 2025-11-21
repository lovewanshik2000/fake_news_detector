# Fake New Detector

A small Flask web app and set of scripts to detect fake news and deepfakes using both text and visual models.

This project contains a text classifier (SVM / Logistic Regression variants) that uses a vectorizer + scikit-learn model, plus a visual deepfake detector implemented as a TensorFlow/Keras model for images and short videos.

## Features

- Web UI (Flask) to:
  - Submit text for fake/real classification.
  - Upload an image or video for deepfake detection.
- Pretrained models saved in the `models/` folder.
- Training and evaluation scripts for text and visual models.
- An HTML report generator with misclassified samples and evaluation images.

## Quick start (development)

1. Create a Python virtual environment (recommended):

```bash
python3 -m venv env
source env/bin/activate     #Linux
.\env\Scripts\activate      #Windos
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app (development mode):

```bash
python app.py
```

By default the app runs on http://127.0.0.1:5000/. Open that in your browser to use the UI (`templates/index.html`).

## Project layout

- `app.py` — Flask web app. Routes:
  - `/` : UI (index page)
  - `/analyze` : POST endpoint for text or file analysis
- `train_models.py` — Top-level script to run training for both visual and text models (script may orchestrate other training scripts).
- `LogisticRegression_train_text_model.py` — Train a logistic regression text model and save vectorizer and model.
- `SVM_train_text_model.py` — Train an SVM text model and save vectorizer and model.
- `check_text_model_accuracy.py` — Evaluate text model accuracy and produce the HTML report in `html_report/`.
- `models/` — Saved model artifacts (example files included):
  - `text_model.joblib`, `text_model_v2.joblib`, `text_model_svm_v3.joblib` (scikit-learn models)
  - `vectorizer.joblib`, `vectorizer_v2.joblib`, `vectorizer_svm_v3.joblib` (feature vectorizers)
  - `visual_model.h5` (TensorFlow/Keras model for images/videos)
- `data/news.csv` — Example dataset used for training/evaluation.
- `html_report/` — Generated HTML report and images (ROC, confusion matrix, etc.).
- `templates/` and `static/` — Flask templates and CSS for the UI.

## How it works (internals)

- Text classification: text is transformed via a saved vectorizer (TF-IDF or similar) and passed to a scikit-learn classifier (SVM or logistic regression). The model's `predict_proba` is used to return a confidence score.
- Visual analysis: the Keras model expects images resized to 224x224 (configured in `app.py`). For videos, the app extracts a small set of frames (default 5) evenly spaced, runs inference on each, and averages the scores.

## Running training & evaluation

- Train text model (example):

```bash
python LogisticRegression_train_text_model.py
# or
python SVM_train_text_model.py
```

Models and vectorizers will be written to `models/` by the training scripts.

- Train visual model: inspect `train_models.py` or other training utilities in the repo. Output model should be saved as `models/visual_model.h5`.

- Generate evaluation report (after running predictions on a holdout set):

```bash
python check_text_model_accuracy.py
# Output: html_report/model_report.html and supporting images/csv
```

## Running the app (API usage)

- Web UI: open `/` in a browser and use the form to submit text or upload a file.
- Programmatic POST (curl example):

```bash
# Text analysis
curl -X POST -F "text_input=Some news text here" http://127.0.0.1:5000/analyze

# Image upload
curl -X POST -F "file_upload=@/path/to/image.jpg" http://127.0.0.1:5000/analyze

# Video upload
curl -X POST -F "file_upload=@/path/to/video.mp4" http://127.0.0.1:5000/analyze
```

The server returns the rendered HTML page. The Flask route internally computes a `result` object with fields like `type`, `message`, `confidence`, and `result_type` which the template displays.

## Dependencies

Dependencies are listed in `requirements.txt`. Key packages include:

- Flask — web app
- numpy, pandas — data handling
- scikit-learn, joblib — text models & serialization
- tensorflow — visual model
- Pillow — image handling
- opencv-python — video/frame extraction

Install with:

```bash
pip install -r requirements.txt
```

## Notes & troubleshooting

- Model files: `app.py` expects models to be in the `models/` directory. If models are missing, start with the training scripts or copy pretrained models to `models/`.
- TensorFlow and Flask: if serving multiple concurrent requests, consider using a production WSGI server (gunicorn) and validate thread-safety with TensorFlow in your environment.
- Video handling: `app.py` writes a temporary file for uploaded videos (and deletes it afterwards). Ensure the process has write permissions to the working directory.
- If `requirements.txt` pins no versions, installing latest packages may introduce incompatibilities; consider pinning working versions if you encounter build/runtime issues.

## Suggested next steps / improvements

- Add a `requirements.lock` or pinned versions for reproducible environments.
- Add unit tests for the model utilities and a small integration test that runs `app.py` and posts sample inputs.
- Add Dockerfile for easy deployment.

## License & attribution

Add a license file if you intend to publish or share (e.g., MIT, Apache-2.0).

---
# fake_news_detector
