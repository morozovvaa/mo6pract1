from flask import Flask, request, jsonify, render_template
import joblib
import gdown
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'wine_secret'

MODEL_PATH = "final_wine_model.pkl"
SCALER_PATH = "scaler.pkl"

# Загрузка модели и скейлера, если не загружены
if not os.path.exists(MODEL_PATH):
    file_id = "1UK4FKJYK4E5_dng6IUmrE0XHFlgtBtd7"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        features = [
            data['fixed_acidity'],
            data['volatile_acidity'],
            data['citric_acid'],
            data['residual_sugar'],
            data['chlorides'],
            data['free_sulfur_dioxide'],
            data['total_sulfur_dioxide'],
            data['density'],
            data['pH'],
            data['sulphates'],
            data['alcohol'],
            data['wine_type']
        ]
        X = np.array([features])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        proba_dict = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(np.max(probabilities)),
            'probabilities': proba_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
