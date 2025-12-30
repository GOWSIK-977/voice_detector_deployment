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

        y, sr = librosa.load(file_path, sr=None, mono=True)

        features = [
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.std(y),
            np.median(y),
            np.percentile(y, 25),
            np.percentile(y, 75),
            np.percentile(y, 75) - np.percentile(y, 25),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_flatness(y=y)),
            np.mean(y),
            np.min(y),
            np.max(y),
            np.ptp(y),
            np.std(y)
        ]

        X = np.array([features])
        prediction = model.predict(X)[0]
        confidence = float(model.predict_proba(X).max())

        os.remove(file_path)

        return jsonify({
            "prediction": {
                "gender": "male" if prediction == 1 else "female",
                "confidence": round(confidence, 3)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
