# chat.py - это основной скрипт для работы чат-бота.
# Он загружает обученную модель (data.pth), обрабатывает
# ввод пользователя и выдаёт ответы на основе предсказаний нейросети.

# ! Импорт необходимых модулей
#   random – для случайного выбора ответа из списка возможных.
#   json – для загрузки JSON-файла с намерениями (intents.json).
#   torch – для работы с нейросетевой моделью (PyTorch).
#   nltk – для обработки естественного языка (токенизация, лемматизация и т. д.).
#   Counter из collections – для хранения статистики частых запросов.
import random
import json
import torch
import nltk

from collections import Counter
# NeuralNet – импорт нейросетевого моделя из файла model.py.
# bag_of_words, tokenize – импорт функций обработки текста из nltk_utils.py.
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# ! Импорт модули telegram для создания телеграм-бота.
#   Update – для получения входящих сообщений.
#   InlineKeyboardButton, InlineKeyboardMarkup – для создания интерактивных кнопок.
#   ReplyKeyboardMarkup – для обычных кнопок под строкой ввода.
#   ApplicationBuilder – для сборки бота.
#   CommandHandler, MessageHandler – обработчики команд и сообщений.
#   TEXT – фильтр, чтобы бот реагировал только на текстовые сообщения.
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes
from telegram.ext.filters import TEXT

# ! Загрузка необходимых данных NLTK:
# punkt – для токенизации текста.
# stopwords – список стоп-слов (бесполезные слова).
# averaged_perceptron_tagger – POS (разметка частей речи).
# wordnet – для лемматизации слов.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# stopwords – список стоп-слов (ненужные слова вроде "и", "но", "или").
# WordNetLemmatizer – для приведения слов к начальной форме (например, "кошки" → "кошка").
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ! Создаем объект лемматизатора для лемматизации слов.
lemmatizer = WordNetLemmatizer()

# ! Функция нормализации текста (удаление стоп-слов, лемматизация, POS tagging)
#   Приводит текст к нижнему регистру и разбивает на слова (word_tokenize).
#   Удаляет знаки препинания.
#   Удаляет стоп-слова.
#   Применяет лемматизацию.
#   Выполняет POS-теггинг (размечает части речи).
#   Возвращает список обработанных слов.
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())  # Токенизация и приведение к нижнему регистру
    tokens = [word for word in tokens if word.isalnum()]  # Удаление знаков препинания
    tokens = [word for word in tokens if word not in stopwords.words("russian")]  # Удаление стоп-слов
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Лемматизация
    pos_tags = nltk.pos_tag(tokens)  # Частеречная разметка (POS tagging)
    return [word for word, tag in pos_tags]

# ! Загрузка модели и данных
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Определяет, использовать ли CUDA (если доступна) или CPU.

# ! Загружает JSON-файл с намерениями (intents).
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# ! Загружает сохранённые параметры модели (data.pth).
FILE = "data.pth"
data = torch.load(FILE)

# ! Извлекает параметры модели из сохранённого файла.
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Создаёт объект нейросетевой модели и загружает её веса (state_dict).
# model.eval() переводит модель в режим предсказания.
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# ! Имя бота и его TOKEN (секретный ключ для доступа к Telegram API).
bot_name = "Pet"
TOKEN = "7999516169:AAEKUaq1we5S9vl4AHYvzazzLTJIx971_Nc"

# ! Статистика повторяющихся запросов
request_counter = Counter() # Счётчик для хранения количества повторяющихся запросов пользователей.

# ! Функция обработки сообщений (Получает текст сообщения от пользователя.)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.lower()

    # ! Если пользователь вводит quit, бот завершает диалог.
    if user_input == "quit":
        await update.message.reply_text("Goodbye!")
        return

    # Проверяем кнопки (Если пользователь нажал на кнопку, вызываем соответствующую функцию.)
    if user_input == "частые вопросы":
        await handle_faq_button(update, context)
        return
    elif user_input == "связаться с ветеринаром":
        await handle_contact_button(update, context)
        return

    # Запоминаем частоту запросов
    request_counter[user_input] += 1 # Увеличиваем счётчик запросов для анализа популярных вопросов.

    # Нормализация текста перед обработкой
    normalized_text = preprocess_text(user_input) # Обрабатываем текст.
    X = bag_of_words(normalized_text, all_words) # и преобразуем его в bag-of-words
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # Получаем предсказанный tag (категорию запроса).
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Вычисляем вероятность предсказания.
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Если вероятность выше 75%, выбираем случайный ответ из intents.json.
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                await update.message.reply_text(response)
                return
    # Иначе бот сообщает, что не понял сообщение.
    else:
        await update.message.reply_text("Извините, я не понимаю...")

# ! Функции кнопок
# Функция для вывода контактной (Создаёт кнопку для перехода в чат с ветеринаром.)
async def handle_contact_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = "buzurg_2003"
    keyboard = [[InlineKeyboardButton("Перейти в чат", url=f"tg://resolve?domain={username}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Свяжитесь с ветеринаром, нажав кнопку ниже:", reply_markup=reply_markup)

# ! Функция для вывода часто задаваемых вопросов
#   Показывает 5 самых популярных запросов.
async def handle_faq_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    most_common = request_counter.most_common(5)
    if most_common:
        common_text = "\n".join([f"{q[0]} ({q[1]} раз)" for q in most_common])
        await update.message.reply_text(f"📌 Часто задаваемые вопросы:\n{common_text}")
    else:
        await update.message.reply_text("Пока нет часто задаваемых вопросов.")

# ! Функция запуска бота с кнопками
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Частые вопросы", "Связаться с ветеринаром"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text("Hello! I'm Pet. How can I help you today?", reply_markup=reply_markup)

# ! Главная функция (Запуск бота)
#   Инициализирует бота и запускает его в режиме polling.
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(TEXT, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
