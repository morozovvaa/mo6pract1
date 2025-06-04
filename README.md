# mo6pract1
 python app.py
 models
 scaler.pkl

https://drive.google.com/file/d/1UK4FKJYK4E5_dng6IUmrE0XHFlgtBtd7/view?usp=drive_link
ссылка на final_wine_model.pkl

GitBash
curl -X POST -H "Content-Type: application/json" \
-d '{"fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0, "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11, "total sulfur dioxide": 34, "density": 0.9978, "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4, "type": 1}' \
http://127.0.0.1:5000/predict

![image](https://github.com/user-attachments/assets/9e7e8ba1-4878-4a96-ac06-1c191f1be3a4)

"prediction": 5 — модель предсказала качество вина = 5.

"probabilities" — список вероятностей для каждого класса (качества от 3 до 9).
Соответственно:

Качество	Вероятность
3	0.0018
4	0.1447
5	0.7587 ← максимальная
6	0.0900
7	0.0048
8	0.0
9	0.0

