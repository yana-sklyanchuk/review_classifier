from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

app = Flask(__name__)

access_token = "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"


# Загрузка обученной модели анализа настроений и токенизатора
model_path = "yana-sklyanchuk/restaurant_reviews"  
model = AutoModelForSequenceClassification.from_pretrained("yana-sklyanchuk/restaurant_reviews", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

@app.route('/classify_reviews', methods=['POST'])
def classify_reviews():
    data = request.get_json()
    reviews = data.get('reviews', [])

    # Проверка наличия отзывов
    if not reviews:
        return jsonify({"error": "No reviews provided"}), 400

    # Предварительная обработка и токенизация отзывов
    inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors='pt', max_length=32)

    # Классификация отзывов
    with torch.no_grad():
        outputs = model(**inputs)

    # Получение прогноза
    predictions = torch.argmax(outputs.logits, dim=1).tolist()

    # Сопоставление прогнозов с метками настроений
    id2label = {0: 'Негативные', 1: 'Позитивные'}
    sentiment_labels = [id2label[prediction] for prediction in predictions]
    
    # Подготовка ответа
    response = {
        'sentiments': sentiment_labels
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
