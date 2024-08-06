from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import sqlite3

app = Flask(__name__)

access_token = "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"


# Load the trained sentiment analysis model and tokenizer
model_path = "yana-sklyanchuk/restaurant_reviews"  # Update this to your model path
model = AutoModelForSequenceClassification.from_pretrained("yana-sklyanchuk/restaurant_reviews", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect('reviews.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classified_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT NOT NULL,
            sentiment TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/classify_reviews', methods=['POST'])
def classify_reviews():
    data = request.get_json()
    reviews = data.get('reviews', [])
    
    # Check if reviews are provided
    if not reviews:
        return jsonify({"error": "No reviews provided"}), 400
    
    # Preprocess and tokenize the reviews
    inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors='pt', max_length=32)
    
    # Classify the reviews
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=1).tolist()
    
    # Map predictions to sentiment labels
    id2label = {0: 'Негативные', 1: 'Позитивные'}  # Adjust labels as needed
    sentiment_labels = [id2label[prediction] for prediction in predictions]
    
    # Save results to the database
    save_to_db(reviews, sentiment_labels)
    
    # Prepare the response
    response = {
        'sentiments': sentiment_labels
    }
    
    return jsonify(response)

def save_to_db(reviews, sentiments):
    conn = sqlite3.connect('reviews.db')
    cursor = conn.cursor()
    for review, sentiment in zip(reviews, sentiments):
        cursor.execute('INSERT INTO classified_reviews (review, sentiment) VALUES (?, ?)', (review, sentiment))
    conn.commit()
    conn.close()

@app.route('/get_classified_reviews', methods=['GET'])
def get_classified_reviews():
    conn = sqlite3.connect('reviews.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM classified_reviews')
    rows = cursor.fetchall()
    conn.close()
    
    # Prepare the response
    classified_reviews = [{"id": row[0], "review": row[1], "sentiment": row[2]} for row in rows]
    return jsonify(classified_reviews)

if __name__ == '__main__':
    app.run(debug=True)

