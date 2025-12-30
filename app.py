import librosa
import numpy as np
from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("random_forest_model.joblib")

@app.route("/recordings", methods=["POST"])
def predict_from_audio():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files["audio"]
        file_path = "temp.wav"
        audio_file.save(file_path)

        # Load audio
        y, sr = librosa.load(file_path, sr=None)

        # Extract same features used during training
        features = {
            "meanfreq": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "sd": np.std(y),
            "median": np.median(y),
            "Q25": np.percentile(y, 25),
            "Q75": np.percentile(y, 75),
            "IQR": np.percentile(y, 75) - np.percentile(y, 25),
            "skew": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "kurt": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "sp.ent": np.mean(librosa.feature.spectral_entropy(y=y)),
            "sfm": np.mean(librosa.feature.spectral_flatness(y=y)),
            "mode": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "meanfun": np.mean(y),
            "minfun": np.min(y),
            "maxfun": np.max(y),
            "meandom": np.mean(y),
            "mindom": np.min(y),
            "maxdom": np.max(y),
            "dfrange": np.ptp(y),
            "modindx": np.std(y)
        }

        X = np.array([list(features.values())])
        prediction = model.predict(X)[0]

        return jsonify({
            "prediction": {
                "gender": "male" if prediction == 1 else "female",
                "confidence": 0.9
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
