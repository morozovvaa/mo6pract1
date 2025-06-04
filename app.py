from flask import Flask, request, jsonify
import joblib
import pandas as pd
import gdown

app = Flask(__name__)

# Загрузим модель и препроцессоры
MODEL_PATH = "final_wine_model.pkl"


# Загружаем модель из облачного хранилища, если её нет локально
file_id = "1UK4FKJYK4E5_dng6IUmrE0XHFlgtBtd7"
url = f"https://drive.google.com/uc?id={file_id}"

output = "final_wine_model.pkl"
gdown.download(url, output, quiet=False)

# Загружаем объекты
model = joblib.load(MODEL_PATH)
scaler = joblib.load("models/scaler.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Получаем JSON с признаками
    input_df = pd.DataFrame([data])

    # Преобразование: заполнение пропусков и масштабирование
    input_scaled = scaler.transform(input_df)

    # Предсказание
    pred_class = model.predict(input_scaled)
    pred_proba = model.predict_proba(input_scaled)

    return jsonify({
        "prediction": int(pred_class[0]),
        "probabilities": pred_proba[0].tolist()
    })


if __name__ == '__main__':
    app.run(debug=True)
