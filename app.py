import librosa
import numpy as np
from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
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

        # ---- FEATURE EXTRACTION (MATCH TRAINING DATA EXACTLY) ----
        meanfreq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        sd = np.std(y)
        median = np.median(y)
        q25 = np.percentile(y, 25)
        q75 = np.percentile(y, 75)
        iqr = q75 - q25
        skew = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        kurt = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        sp_ent = np.mean(librosa.feature.spectral_entropy(y=y))
        sfm = np.mean(librosa.feature.spectral_flatness(y=y))
        mode = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        meanfun = np.mean(y)
        minfun = np.min(y)
        maxfun = np.max(y)
        meandom = np.mean(y)
        mindom = np.min(y)
        maxdom = np.max(y)
        dfrange = np.ptp(y)
        modindx = np.std(y)

        features = [[
            meanfreq, sd, median, q25, q75, iqr, skew, kurt,
            sp_ent, sfm, mode, centroid,
            meanfun, minfun, maxfun,
            meandom, mindom, maxdom,
            dfrange, modindx
        ]]

        prediction = model.predict(features)[0]
        confidence = float(model.predict_proba(features).max())

        os.remove(file_path)

        return jsonify({
            "prediction": "male" if prediction == 1 else "female",
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
