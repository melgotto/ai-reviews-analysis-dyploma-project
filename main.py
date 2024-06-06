#A File with testing the main functionalities of project

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#tokens = tokenizer.encode('поганий екран, але непогана начинка та чудовий досвід використання', return_tensors='pt')

#result = model(tokens)

#print(int(torch.argmax(result.logits))+1)

#soup = BeautifulSoup(r.text, 'html.parser')
#regex = re.compile('.*comment.*')
#results = soup.find_all('p', {'class': regex})
#reviews = [result.text for result in results] !!!!!!!!!!!!!!!!!product-comments__list-item


def find_comment_classes(html_content):
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all elements with class attribute
    elements_with_class = soup.find_all(class_=True)

    # Dictionary to store classes containing 'comment' and their corresponding text content
    comment_classes_text = {}

    # Iterate through elements with class attribute
    for element in elements_with_class:
        classes = element['class']  # Get classes of the element
        text_content = element.get_text(strip=True)  # Get text content of the element
        for class_name in classes:
            if 'product-comments__list-item' in class_name:  # Check if 'comment' is in class name
                if class_name not in comment_classes_text:
                    comment_classes_text[class_name] = []  # Initialize list if class name not already in dictionary
                comment_classes_text[class_name].append(text_content)


    return comment_classes_text

url = 'https://rozetka.com.ua/ua/puma_4064536587899_eu/p357787365/comments/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}

r = requests.get(url, headers=headers)
h = r.text

comments = find_comment_classes(h)
for class_name, text_content_list in comments.items():
    print(f"Class '{class_name}' content:")
    for text_content in text_content_list:
        print("  -", text_content)
        # Check if the length of text content is less than 512
        if len(text_content) < 512:
            tokens = tokenizer.encode(text_content, return_tensors='pt')
            result = model(tokens)
            print(int(torch.argmax(result.logits)) + 1)
        else:
            print("Text length exceeds 512, skipping sentiment analysis.")
