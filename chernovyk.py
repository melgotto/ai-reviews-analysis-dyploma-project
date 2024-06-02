from transformers import pipeline

# Завантажуємо попередньо навчену модель для класифікації настрою
classifier = pipeline('sentiment-analysis')

# Текст для класифікації
text = "I hate using Hugging Face Transformers!"

# Виконуємо класифікацію
result = classifier(text)

print(result)