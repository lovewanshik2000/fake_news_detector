import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer

# =======================================================
# CONFIGURATION
# =======================================================
MODEL_PATH = "models/text_model_svm_v3.joblib"
VECTORIZER_PATH = "models/vectorizer_svm_v3.joblib"
DATA_PATH = "data/news.csv"

REPORT_DIR = "html_report"
IMAGES_DIR = os.path.join(REPORT_DIR, "images")
REPORT_FILE = os.path.join(REPORT_DIR, "model_report.html")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# =======================================================
# LOAD MODEL
# =======================================================
print("Loading model...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# =======================================================
# LOAD & FIX DATASET
# =======================================================
print("Cleaning dataset...")

df = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")

# Fix extra columns caused by malformed CSV rows
if "text" not in df.columns or "label" not in df.columns:
    df = df.iloc[:, :2]
    df.columns = ["text", "label"]

# Clean text & labels
df["text"] = df["text"].astype(str).str.replace("\t", " ", regex=False).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.upper()

# Keep only REAL/FAKE
df = df[df["label"].isin(["REAL", "FAKE"])]

# Remove blank rows
df = df[df["text"] != ""]

X = df["text"]
y_true = df["label"]

# Vectorize
X_vec = vectorizer.transform(X)

# Predictions
proba = model.predict_proba(X_vec)
y_pred = model.predict(X_vec)

# Classes
classes = model.classes_
label_to_idx = {c: i for i, c in enumerate(classes)}
y_true_idx = np.array([label_to_idx[l] for l in y_true])

# =======================================================
# METRICS
# =======================================================
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
report_text = classification_report(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)

# =======================================================
# VISUALIZATIONS
# =======================================================

# 1. Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
cm_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# 2. Probability Distribution
plt.figure(figsize=(6, 5))
plt.hist(proba[:, 1], bins=20, alpha=0.7)
plt.title("Probability Distribution (Fake Class)")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
prob_path = os.path.join(IMAGES_DIR, "probability_distribution.png")
plt.savefig(prob_path)
plt.close()

# 3. ROC Curve
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(y_true).ravel()

fpr, tpr, _ = roc_curve(y_true_bin, proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
roc_path = os.path.join(IMAGES_DIR, "roc_curve.png")
plt.savefig(roc_path)
plt.close()

# 4. Calibration Curve
prob_true, prob_pred = calibration_curve(y_true_bin, proba[:, 1], n_bins=10)

plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("Calibration Curve")
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.legend()
cal_path = os.path.join(IMAGES_DIR, "calibration_curve.png")
plt.savefig(cal_path)
plt.close()

# 5. Misclassified samples
misclassified = df[y_true != y_pred]
mis_path = os.path.join(REPORT_DIR, "misclassified_samples.csv")
misclassified.to_csv(mis_path, index=False)

# =======================================================
# BUILD HTML REPORT
# =======================================================

html = f"""
<html>
<head>
    <title>Text Model Evaluation Report</title>
    <style>
        body {{
            font-family: Arial;
            padding: 20px;
            background: #f7f7f7;
        }}
        .card {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
            border-radius: 8px;
        }}
        pre {{
            background: #1e1e1e;
            color: white;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        img {{
            max-width: 100%;
            border-radius: 6px;
        }}
    </style>
</head>
<body>

<h1>📰 Fake News Classifier — Evaluation Report</h1>
<p><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="card">
<h2>📊 Summary Metrics</h2>
<p><b>Accuracy:</b> {accuracy*100:.2f}%<br>
<b>Precision:</b> {precision:.3f}<br>
<b>Recall:</b> {recall:.3f}<br>
<b>F1 Score:</b> {f1:.3f}<br>
<b>AUC:</b> {roc_auc:.3f}</p>
</div>

<div class="card">
<h2>📘 Classification Report</h2>
<pre>{report_text}</pre>
</div>

<div class="card">
<h2>📌 Confusion Matrix</h2>
<img src="images/confusion_matrix.png">
</div>

<div class="card">
<h2>📈 ROC Curve</h2>
<img src="images/roc_curve.png">
</div>

<div class="card">
<h2>🎯 Calibration Curve</h2>
<img src="images/calibration_curve.png">
</div>

<div class="card">
<h2>📊 Probability Distribution</h2>
<img src="images/probability_distribution.png">
</div>

<div class="card">
<h2>❗ Misclassified Samples</h2>
<p>See file: {mis_path}</p>
<pre>{misclassified.head(10).to_string()}</pre>
</div>

</body>
</html>
"""

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print("\n✔ HTML Report Generated:", REPORT_FILE)

# =======================================================
# OPEN IN BROWSER
# =======================================================
print("Opening report in browser...")
webbrowser.open("file://" + os.path.abspath(REPORT_FILE))
