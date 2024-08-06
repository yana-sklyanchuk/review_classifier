# [Модель генерации отзывов о ресторане](generative_model.ipynb)
Проект направлен на создание модели, способной генерировать отзывы о ресторанах с помощью доработанной версии модели GPT-2. Модель обучается на наборе данных существующих отзывов, что позволяет ей изучить структуру и стиль отзывов о ресторанах.
## Подготовка данных
- Загрузка предварительно обученной модели и токенизатора: Токенизатор и модель GPT-2 загружаются с помощью библиотеки transformers.
- Загрузка набор данных: Набор данных загружается из текстового файла (reviews.txt) с помощью класса TextDataset, который подготавливает данные для языкового моделирования.
- Установка размера блока: Размер блока для набора данных установлен на 128 лексем, который может быть изменен в зависимости от конкретных потребностей задачи.
## Обучение модели
- Создание коллатора данных: Коллатор данных для языкового моделирования создается с помощью DataCollatorForLanguageModeling, при этом масочное моделирование языка отключено.
- Определение аргументов для обучения: Настраиваются аргументы для обучения, в том числе:
1. Output directory for the model.
2. Number of epochs (set to 3).
3. Batch size (set to 2).
4. Steps for saving the model.
5. Option to push the model to the Hugging Face Hub.
- Инициализация Trainer: Создается экземпляр Trainer, которому передаются модель, аргументы для обучения, коллатор данных и набор данных для обучения.
- Обучение модели: Модель обучается с помощью метода trainer.train().
## Развертывание модели
После обучения модель и токенизатор размещаются на Hugging Face Hub для легкого доступа и совместного использования.
## Оценка модели
Чтобы оценить эффективность модели, настроенная модель загружается из Hugging Face Hub, и задается подсказка для создания отзывов о ресторане. Модель генерирует текст на основе подсказки, причем для разнообразия можно создавать несколько последовательностей.
## Пример генерации текста
Для генерации используется подсказка: "В ресторане был". Модель генерирует три разных отзыва на основе этого запроса:
```python
from transformers import pipeline

access_token = "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"
model_name = "yana-sklyanchuk/dialogue_reviews" 

# Load the fine-tuned model and tokenizer from the Hugging Face Hub
generator = pipeline('text-generation', model=model_name, token = access_token)
# Define a prompt for text generation
prompt = "The restaurant had"
# Generate text based on the prompt
generated_texts = generator(prompt, max_length=50, num_return_sequences=3)
# Print the generated texts
for i, text in enumerate(generated_texts):
    print(f"Generated Text {i + 1}: {text['generated_text']}")
```
Ожидаемый ответ:
```
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Generated Text 1: The restaurant had a lovely menu, from chicken wing soup to lobster fries to shrimp burgers!
They have some classic dishes in place including our pork tenderloin, pizza style, hot sauce, crispy chicken fries to rave reviews.
It's a
Generated Text 2: The restaurant had a lot of food and very clean ambience I think.
I loved this place.
I won't be back.
What a deal.
The ambiance is perfect.
I'm very much on the fence about staying
Generated Text 3: The restaurant had an amazing atmosphere and service.
They got you a $4 lunch worth of food, as well!
The waitress was kind to us.
The food tasted good even though there was just over a pound missing from it.
```