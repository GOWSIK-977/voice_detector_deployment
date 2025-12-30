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
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Feature extraction (SAFE & STABLE)
        features = [
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),  # meanfreq
            np.std(y),                                                # sd
            np.median(y),                                             # median
            np.percentile(y, 25),                                     # Q25
            np.percentile(y, 75),                                     # Q75
            np.percentile(y, 75) - np.percentile(y, 25),              # IQR
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),    # rolloff
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),  # bandwidth
            np.mean(librosa.feature.spectral_flatness(y=y)),          # flatness
            np.mean(y),                                               # meanfun
            np.min(y),                                                # minfun
            np.max(y),                                                # maxfun
            np.ptp(y),                                                # dfrange
            np.std(y)                                                 # modindx
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
