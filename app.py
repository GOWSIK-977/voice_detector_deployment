import librosa
import numpy as np
from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS
from scipy.stats import skew, kurtosis

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

        # ---- FEATURE EXTRACTION ----
        meanfreq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        sd = np.std(y)
        median = np.median(y)
        q25 = np.percentile(y, 25)
        q75 = np.percentile(y, 75)
        iqr = q75 - q25
        skew_val = skew(y)
        kurt_val = kurtosis(y)

        # Spectral Entropy manually
        S = np.abs(librosa.stft(y))**2
        ps = S / np.sum(S, axis=0, keepdims=True)
        sp_ent = np.mean(-np.sum(ps * np.log2(ps + 1e-10), axis=0))

        sfm = np.mean(librosa.feature.spectral_flatness(y=y))
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
            meanfreq, sd, median, q25, q75, iqr,
            skew_val, kurt_val, sp_ent, sfm,
            centroid, meanfun, minfun, maxfun,
            meandom, mindom, maxdom, dfrange, modindx
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
