from flask import Flask, render_template, request, jsonify
import joblib 
import numpy as np
import tensorflow as tf 
from PIL import Image
import os
import io
import time # unique file naming 
import cv2 

# TensorFlow Thread Safety (Best Practice for Flask)
tf.keras.backend.clear_session()

app = Flask(__name__)


# --- Configuration Paths ---
TEXT_MODEL_PATH = 'models/text_model.joblib'
VECTORIZER_PATH = 'models/vectorizer.joblib'
VISUAL_MODEL_PATH = 'models/visual_model.h5'
IMG_SIZE = (224, 224) # Defined here for clarity

# --- Global Model Loading ---
text_model = None
vectorizer = None
visual_model = None
models_loaded = False

try:
    # Load the text classification model and vectorizer
    text_model = joblib.load(TEXT_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Load the visual deepfake model
    visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH, compile=False)
    
    models_loaded = True
    print("✅ All models loaded successfully.")
except FileNotFoundError:
    print(f"⚠️ Warning: Model files not found in the 'models' folder. Please run train_models.py first.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# --- Utility Functions ---

def analyze_text(text):
    """
    Analyzes a given text string for fake news and returns probability scores.
    """
    if not models_loaded or text_model is None or vectorizer is None:
        return {
            "type": "Error",
            "message": "Text analysis models are not loaded. Please check the 'models' directory.",
            "confidence": 0,
            "result_type": "Error"
        }

    try:
        # 1. Convert text to numerical features using the vectorizer
        text_vec = vectorizer.transform([text])
        
        # 2. Get probabilities for all classes (e.g., [Prob_REAL, Prob_FAKE])
        probabilities = text_model.predict_proba(text_vec)[0]
        classes = text_model.classes_ 

        # 3. Determine the predicted class and its confidence
        max_prob_index = np.argmax(probabilities)
        predicted_label = classes[max_prob_index]
        confidence_score = probabilities[max_prob_index] * 100
        
        # 4. Formulate the result
        result_type = 'REAL' if predicted_label == 'REAL' else 'FAKE'
        message = f"This text is likely **{result_type}** news."

        return {
            "type": "Text Analysis",
            "message": message,
            "confidence": round(confidence_score, 2),
            "percentage": round(confidence_score, 2), # Duplicate for clarity in UI requirement
            "result_type": result_type # 'FAKE' or 'REAL'
        }
    except Exception as e:
        return {
            "type": "Error",
            "message": f"Error during text analysis: {e}",
            "confidence": 0,
            "result_type": "Error"
        }


def analyze_visual(file_stream, file_type):
    """
    Analyzes an image or video for deepfake signs.
    FIXED: Improved confidence calculation for REAL results.
    """
    if not models_loaded or visual_model is None:
        return {
            "type": "Error",
            "message": "Visual analysis model not loaded. Please check the 'models' directory.",
            "confidence": 0,
            "result_type": "Error"
        }
    
    if file_type.startswith('image/'):
        try:
            # Image processing logic
            img = Image.open(file_stream).convert('RGB')
            img = img.resize(IMG_SIZE)
            # Normalization must match training (1./255)
            img_array = np.array(img) / 255.0 
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction returns a single value between 0 and 1 (Deepfake/Fake score)
            prediction = visual_model.predict(img_array, verbose=0)
            deepfake_score = prediction[0][0]
            
            # --- Confidence Calculation Logic ---
            if deepfake_score >= 0.5:
                # Prediction leans towards Fake (1)
                confidence = round(deepfake_score * 100, 2)
                result_type = 'FAKE'
                message = f"The image is likely **Fake** (Deepfake/AI-generated)."
            else:
                # Prediction leans towards Real (0)
                # Authentic score is (1 - deepfake_score)
                authentic_score = 1 - deepfake_score
                confidence = round(authentic_score * 100, 2)
                result_type = 'REAL'
                message = f"The image is likely **Real** (Authentic)."
            # --- End Confidence Calculation Logic ---
                
            return {
                "type": "Image Analysis",
                "message": message,
                "confidence": confidence,
                "percentage": confidence,
                "result_type": result_type
            }
        except Exception as e:
            return {
                "type": "Error",
                "message": f"Could not process image: {e}",
                "confidence": 0,
                "result_type": "Error"
            }
    
    elif file_type.startswith('video/'):
        unique_id = int(time.time() * 1000)
        temp_file_path = f"temp_video_{unique_id}.mp4" 
        
        try:
            # 1. Save the file stream to a temporary file for OpenCV
            with open(temp_file_path, "wb") as f:
                f.write(file_stream.getbuffer())

            # 2. Video analysis logic
            cap = cv2.VideoCapture(temp_file_path)
            frames_to_analyze = 5 
            deepfake_scores = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip_interval = max(1, total_frames // frames_to_analyze)

            for i in range(frames_to_analyze):
                frame_idx = i * skip_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Preprocessing
                frame_resized = cv2.resize(frame, IMG_SIZE)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_normalized = np.expand_dims(frame_rgb, axis=0) / 255.0
                
                # Prediction
                prediction = visual_model.predict(frame_normalized, verbose=0)
                deepfake_scores.append(prediction[0][0])
                
            cap.release()
            
            if not deepfake_scores:
                return {
                    "type": "Video Analysis",
                    "message": "Error: Could not extract readable frames from video.",
                    "confidence": 0,
                    "result_type": "Error"
                }
                    
            # 3. Average the deepfake scores across the analyzed frames
            avg_deepfake_score = np.mean(deepfake_scores)
            
            # --- Confidence Calculation Logic for Video ---
            if avg_deepfake_score >= 0.5:
                # Prediction leans towards Fake (1)
                confidence = round(avg_deepfake_score * 100, 2)
                result_type = 'FAKE'
                message = f"This video is likely **Fake** (Deepfake/AI-generated)."
            else:
                # Prediction leans towards Real (0)
                authentic_confidence = round((1 - avg_deepfake_score) * 100, 2)
                confidence = authentic_confidence
                result_type = 'REAL'
                message = f"This video is likely **Real** (Authentic)."
            # --- End Confidence Calculation Logic for Video ---
            
            return {
                "type": "Video Analysis",
                "message": message,
                "confidence": confidence,
                "percentage": confidence,
                "result_type": result_type
            }
        except Exception as e:
            return {
                "type": "Error",
                "message": f"Could not process video: {e}",
                "confidence": 0,
                "result_type": "Error"
            }
        finally:
            # 4. Clean up temporary file in all cases
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    return {
        "type": "Unknown",
        "message": "Unsupported file type or no input provided.",
        "confidence": 0,
        "result_type": "Error"
    }

# --- Flask Routes ---

@app.route('/')
def home():
    # Initialize result_data for the template context
    result_data = {
        'type': 'Welcome',
        'message': 'Enter text or upload an image/video to start analysis.',
        'confidence': 0,
        'percentage': 0,
        'result_type': 'Pending'
    }
    return render_template('index.html', result=result_data, models_loaded=models_loaded)

@app.route('/analyze', methods=['POST'])
def analyze():
    result = None
    text = request.form.get('text_input', '').strip()
    file_upload = request.files.get('file_upload')

    # Prioritize text input if provided, otherwise check file upload
    if text:
        result = analyze_text(text)
    
    elif file_upload and file_upload.filename:
        # Read the file data into an in-memory stream (BytesIO)
        file_stream = io.BytesIO(file_upload.read())
        file_type = file_upload.mimetype
        result = analyze_visual(file_stream, file_type)

    # If neither input was valid
    if result is None:
        result = {
            'type': 'Warning', 
            'message': 'Please enter text or select a file for analysis.',
            'confidence': 0,
            'percentage': 0,
            'result_type': 'Error'
        }

    return render_template('index.html', result=result, models_loaded=models_loaded)

if __name__ == '__main__':
    # Ensure the models directory exists for saving/loading
    if not os.path.exists('models'):
        os.makedirs('models')
    
    app.run(debug=True)
