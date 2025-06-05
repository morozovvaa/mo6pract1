# Практическая работа 1: Машинное обучение для предсказания качества вина
Ссылки на ресурсы
Jupyter Notebook 
Kaggle https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
Google Colab https://colab.research.google.com/drive/10eN7-q0CUsLe_n6GGkjQjcdujm1vMDSo?usp=sharing

Развернутые сервисы
JSON API сервис
Веб-приложение

Введение
Целью данного проекта является построение модели классификации, способной предсказывать качество вина на основе его химических характеристик. Используются два датасета:

winequality-red.csv — данные о красном вине
winequality-white.csv — данные о белом вине
Качество вина представлено целевой переменной quality, принимающей значения от 3 до 8.

Данный проект представляет собой комплексное исследование в области машинного обучения, направленное на решение задачи классификации качества вина на основе его химических характеристик. Проект включает в себя полный цикл разработки: от исследовательского анализа данных до развертывания готового решения в виде веб-сервиса.

Датасет и цели исследования
Описание датасета
Источник: Wine Quality Dataset

Датасет содержит информацию о красных и белых винах португальского региона "Винью Верде". Каждое вино описывается 11 физико-химическими характеристиками и имеет оценку качества от экспертов по шкале от 0 до 10.

Характеристики вина:
fixed acidity - фиксированная кислотность (г/л)
volatile acidity - летучая кислотность (г/л)
citric acid - лимонная кислота (г/л)
residual sugar - остаточный сахар (г/л)
chlorides - хлориды (г/л)
free sulfur dioxide - свободный диоксид серы (мг/л)
total sulfur dioxide - общий диоксид серы (мг/л)
density - плотность (г/мл)
pH - уровень pH
sulphates - сульфаты (г/л)
alcohol - содержание алкоголя (%)
Целевая переменная:
quality - оценка качества (3-9 баллов)
Статистика датасета:
Красное вино: 1599 образцов
Белое вино: 4898 образцов
Общий размер: 6497 образцов

Загрузка и подготовка данных
```
!wget -O winequality.zip "https://drive.google.com/uc?id=1YHDSDHi-Av312W3GHnMclmLJ2dNKQ7k1 " --no-check-certificate
!unzip -q winequality.zip -d wine_data
```
Объединение датасетов
```
df_red = pd.read_csv("wine_data/winequality-red.csv", sep=';')
df_white = pd.read_csv("wine_data/winequality-white.csv", sep=';')

df_red['type'] = 'red'
df_white['type'] = 'white'

df = pd.concat([df_red, df_white], ignore_index=True)
```
Предобработка
```
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1765, stratify=y_trainval, random_state=42)
```
Распределение качества
```
sns.countplot(data=df, x='quality', hue='type')
plt.title('Quality Distribution by Wine Type')
plt.show()
```
Корреляционная матрица
```
corr = df.drop(columns=['type']).corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```
Построение моделей машинного обучения
DummyClassifier
```
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_val)
print(classification_report(y_val, dummy_pred))
```
Логистическая регрессия
```
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_val)
print(classification_report(y_val, lr_pred))
```
Случайный лес
```
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
print(classification_report(y_val, rf_pred))
```
XGBoost
```
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train_enc)
xgb_pred = xgb.predict(X_val)
print(classification_report(y_val_enc, xgb_pred))
```
Нейронная сеть
```
model = Sequential()
model.add(Dense(100, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), metrics=['accuracy'], loss='categorical_crossentropy')
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=500, batch_size=32)
```
Сравнение моделей
Лучшая модель:
Random Forest
Финальная модель и сохранение
Для финальной модели был выбран Random Forest с гиперпараметрами:
```
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
```
Модель обучена на масштабированных данных:
```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

rf.fit(X_train_scaled, y_train)
```
Оценка на тестовой выборке:
```
rf_pred = rf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Macro F1-score:", f1_score(y_test, rf_pred, average='macro'))
```
Сохранение модели
```
joblib.dump(rf, "final_wine_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```



REST API
Сервис написан с использованием Flask , предоставляет один эндпоинт /api/predict , который принимает данные о вине и возвращает результат предсказания в формате JSON.
Выбранная модель
Была обучена и выбрана модель Random Forest Classifier , показавшая наилучшие метрики на тестовой выборке
Модель сохранена в файл final_wine_model.pkl.
Помимо самого класса (качество), модель возвращает вероятности принадлежности к каждому классу (от 3 до 9). Это позволяет не только получить предсказание, но и понять уверенность модели.

Архитектура сервиса
Python
Flask — для создания REST API
joblib — для загрузки обученной модели и скалера
JSON — формат передачи данных
POST -запросы — для отправки данных о вине

Эндпоинты
POST /api/predict
Описание: Принимает параметры вина и возвращает предсказание качества.

Тело запроса (JSON):
```
{
  "fixed acidity": 7.4,
  "volatile acidity": 0.7,
  "citric acid": 0,
  "residual sugar": 1.9,
  "chlorides": 0.076,
  "free sulfur dioxide": 11,
  "total sulfur dioxide": 34,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "type": 1
}
```
Ответ (JSON):
```
{
  "prediction": 5,
  "probabilities": [0.0018, 0.1447, 0.7587, 0.0900, 0.0048, 0.0, 0.0]
}
```
![image](https://github.com/user-attachments/assets/9e7e8ba1-4878-4a96-ac06-1c191f1be3a4)
Тестирование API
Запрос через curl :
```
curl -X POST -H "Content-Type: application/json" \
-d '{"fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0, "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11, "total sulfur dioxide": 34, "density": 0.9978, "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4, "type": 1}' \
http://127.0.0.1:5000/predict
```
Структура проекта
```
wine-quality-api/
├── app.py                  # Flask-приложение
├── final_wine_model.pkl    # Сохранённая модель
├── scaler.pkl              # Сохранённый скалер
├── requirements.txt        # Зависимости
└── README.md               # Описание проекта
```
Установка и запуск
Установите зависимости:
```
pip install -r requirements.txt
```
Запустите сервер
```
python app.py
```
Сервис будет доступен по адресу: http://localhost:5000/

Веб-приложение

В данном проекте была создана интерактивная веб-платформа , которая позволяет пользователям вводить характеристики вина и получать предсказание его качества на основе обученной машинной модели. Приложение также предоставляет визуализацию вероятностей принадлежности к каждому классу качества.
Структура проекта
```
wine_api/
├── app.py                # Flask-приложение
├── templates/
│   └── index.html        # HTML-шаблон интерфейса
├── venv/                 # Виртуальное окружение
├── final_wine_model.pkl  # Сохранённая модель Random Forest
├── scaler.pkl            # Сохранённый скалер
└── requirements.txt       # Зависимости проекта
```
Техническое описание
Backend (Flask)
Основные компоненты:
Загрузка модели и скейлера:
Модель (final_wine_model.pkl) и скейлер (scaler.pkl) загружаются при старте приложения.
Если файлы отсутствуют, они автоматически скачиваются из Google Drive.
```
if not os.path.exists(MODEL_PATH):
    file_id = "1UK4FKJYK4E5_dng6IUmrE0XHFlgtBtd7"
    url = f"https://drive.google.com/uc?id= {file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)
```
Эндпоинт /api/predict:
Принимает POST-запрос с данными о вине в формате JSON.
Производит масштабирование данных с помощью scaler.
Выполняет предсказание с помощью модели.
Возвращает результат в формате JSON
```
{
    "prediction": 5,
    "confidence": 0.7587,
    "probabilities": {
        "3": 0.0018,
        "4": 0.1447,
        "5": 0.7587,
        "6": 0.0900,
        "7": 0.0048,
        "8": 0.0,
        "9": 0.0
    }
}
```
```
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
```
Главная страница (/):
Предоставляет HTML-интерфейс для ввода данных о вине.
```
@app.route('/')
def index():
    return render_template('index.html')
```
Frontend (HTML + JavaScript)
Основные компоненты:
Форма ввода данных:
Пользователь может ввести значения всех необходимых характеристик вина.
Используется Bootstrap для стилизации формы.
```
<form id="prediction-form">
    <div class="feature-input">
        <label for="wine_type">Тип вина</label>
        <select id="wine_type" required>
            <option value="0">Белое</option>
            <option value="1">Красное</option>
        </select>
    </div>
    <div class="feature-input">
        <label for="fixed_acidity">Фиксированная кислотность</label>
        <input type="number" id="fixed_acidity" step="0.1" value="7.0" required>
    </div>
    <!-- Другие поля -->
    <button type="submit">Предсказать качество</button>
</form>
```
Обработка отправки формы:
При отправке формы данные сериализуются в JSON и отправляются на эндпоинт /api/predict.
Результат обрабатывается JavaScript'ом и отображается на странице.
```
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = {
        fixed_acidity: parseFloat(document.getElementById('fixed_acidity').value),
        volatile_acidity: parseFloat(document.getElementById('volatile_acidity').value),
        // Другие поля
        wine_type: parseInt(document.getElementById('wine_type').value)
    };
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    });
    const result = await response.json();
    // Обработка результата
});
```
Визуализация результатов:
Используется библиотека Chart.js для создания графика вероятностей.
После получения ответа от сервера данные передаются в Chart.js для отрисовки.
```
new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['3', '4', '5', '6', '7', '8', '9'],
        datasets: [{
            label: 'Вероятности (%)',
            data: probabilities.map(p => p * 100),
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});
```
Функциональность приложения
Главные возможности:
Ввод данных о вине:
Пользователь может ввести все необходимые характеристики вина через удобную форму.
Предсказание качества:
После отправки формы данные обрабатываются на сервере, и модель выдаёт предсказание.
Отображение результатов:
На экране показывается:
Предсказанное качество вина.
Уровень уверенности модели.
График распределения вероятностей по всем классам.
Интерактивность:
Пользователь может менять параметры в реальном времени и видеть, как меняется результат.
6. Развертывание и запуск
Шаги для запуска:
Создание виртуального окружения
```
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS/Linux
```
Установка зависимостей
```
pip install -r requirements.txt
```
Запуск приложения
```
python app.py
```
Доступ к приложению:
Откройте браузер и перейдите по адресу: http://localhost:5000/

![image](https://github.com/user-attachments/assets/e71604d5-fc97-404f-826c-fdb90bba04d0)





https://drive.google.com/file/d/1UK4FKJYK4E5_dng6IUmrE0XHFlgtBtd7/view?usp=drive_link
ссылка на final_wine_model.pkl





![image](https://github.com/user-attachments/assets/a4f1db31-f6fd-4432-9c15-175f1a89f296)
