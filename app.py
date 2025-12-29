from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("random_forest_model.joblib")

# Initialize Flask app


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Feature names (must match training data)
        feature_names = [
            'meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
            'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun',
            'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom',
            'dfrange', 'modindx'
        ]

        # Convert input to DataFrame
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format"}), 400

        # Ensure correct column order
        input_df = input_df[feature_names]

        # Make prediction
        prediction = model.predict(input_df)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
