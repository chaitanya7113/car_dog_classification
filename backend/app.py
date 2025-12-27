from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import os




# =========================
# APP SETUP
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# CONFIG & MODEL LOADING
# =========================
MODEL_PATH = "final_cat_dog_model.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Cat", "Dog"]

# Load model once at startup
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "Cat vs Dog Prediction API is running 🚀"
    })

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not initialized on server"}), 500

    # 2. Validate input
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # 3. Preprocess Image
        # We read the file stream directly to save memory/time
        img = image.load_img(io.BytesIO(file.read()), target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        
        # Normalization (Ensure this matches your training: 1./255)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Make Prediction
        # For binary classification (1 neuron output with Sigmoid)
        prediction_raw = model.predict(img_array)[0][0]

        # 5. Logic: Usually 0 = Cat, 1 = Dog
        if prediction_raw >= 0.5:
            label = CLASS_NAMES[1]  # Dog
            confidence = float(prediction_raw)
        else:
            label = CLASS_NAMES[0]  # Cat
            confidence = float(1 - prediction_raw)

        return jsonify({
            "success": True,
            "label": label,
            "confidence": round(confidence, 4),
            "raw_value": float(prediction_raw)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    # Use debug=False in production environment
    app.run(host="0.0.0.0", port=5000, debug=True)