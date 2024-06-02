import torch
import re
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Завантаження моделі та токенізатора
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def find_comment_classes(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    elements_with_class = soup.find_all(class_=True)
    comment_classes_text = {}
    for element in elements_with_class:
        classes = element['class']
        text_content = element.get_text(strip=True)
        for class_name in classes:
            if 'product-comments__list-item' in class_name:
                if class_name not in comment_classes_text:
                    comment_classes_text[class_name] = []
                comment_classes_text[class_name].append(text_content)
    return comment_classes_text

def preprocess_text(text):
    # Видалення непотрібного тексту
    text = re.sub(r'\d{2} \w+ \d{4}Відгук від покупця\.Продавець: .*?\.Розмір: .*?\.Загальне враженняРекомендуєте даний товар\?Відповідність фото', '', text)
    return text.strip()

def extract_nickname(text):
    # Виділення нікнейму відправника
    match = re.search(r'^(.*?\s?\d{1,2}\s?\w+)\s\d{4}Відгук від покупця', text)
    if match:
        return match.group(1).strip()
    return "Невідомий"

def analyze_sentiment(review_url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
    response = requests.get(review_url, headers=headers)
    comments = find_comment_classes(response.text)
    sentiments = []
    for class_name, text_content_list in comments.items():
        for text_content in text_content_list:
            nickname = extract_nickname(text_content)
            clean_text = preprocess_text(text_content)
            if len(clean_text) < 512:
                tokens = tokenizer.encode(clean_text, return_tensors='pt')
                result = model(tokens)
                sentiment = torch.argmax(result.logits).item()
                if sentiment == 0:
                    sentiment_text = 'Негативний'
                elif sentiment == 1:
                    sentiment_text = 'Нейтральний'
                else:
                    sentiment_text = 'Позитивний'
                sentiments.append({'nickname': nickname, 'text': clean_text, 'sentiment': sentiment_text})
            else:
                sentiments.append({'nickname': nickname, 'text': clean_text, 'sentiment': 'Текст перевищує 512 символів, аналіз не виконується.'})
    return sentiments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    review_link = data['reviewLink']
    sentiments = analyze_sentiment(review_link)
    return jsonify({'results': sentiments})

if __name__ == '__main__':
    app.run(debug=True)



