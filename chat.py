import random
import json
import torch
import nltk
import openai
import os
import time
import datetime

from collections import Counter
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.ext.filters import TEXT

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from telegram import BotCommand


# API ключ для OpenAI
api_key = "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"

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

# Глобальная переменная для хранения счетчика вопросов
request_counter = Counter()


# Загрузка интентов
def load_intents():
    with open("intents.json", "r", encoding="utf-8") as file:
        return json.load(file)


def save_intents(intents):
    with open("intents.json", "w", encoding="utf-8") as file:
        json.dump(intents, file, ensure_ascii=False, indent=4)


intents = load_intents()


# GPT-4o API вызов
def chat_with_gpt(prompt):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content
    except openai.RateLimitError as e:
        return "Превышен лимит запросов. Попробуйте позже."


# Обработка текста
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words("russian")]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Telegram bot token
TOKEN = "7999516169:AAEKUaq1we5S9vl4AHYvzazzLTJIx971_Nc"


# 🔹 Обработчик кнопки "Связаться с ветеринаром"
async def handle_contact_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = "buzurg_2003"
    keyboard = [[InlineKeyboardButton("Перейти в чат", url=f"tg://resolve?domain={username}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Свяжитесь с ветеринаром, нажав кнопку ниже:", reply_markup=reply_markup)


# 🔹 Обработчик кнопки "Частые вопросы"
async def handle_faq_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    most_common = request_counter.most_common(5)
    if most_common:
        common_text = "\n".join([f"🔹 {q[0]} ({q[1]} раз)" for q in most_common])
        await update.message.reply_text(f"📌 Часто задаваемые вопросы:\n{common_text}")
    else:
        await update.message.reply_text("Пока нет часто задаваемых вопросов.")


# 🔹 Обработчик сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.lower()

    # 🛑 Сначала проверяем, нажал ли пользователь кнопку
    if user_input == "частые вопросы":
        await handle_faq_button(update, context)
        return
    elif user_input == "связаться с ветеринаром":
        await handle_contact_button(update, context)
        return

    # 🔍 Проверяем intents.json
    for intent in intents["intents"]:
        if user_input in [pattern.lower() for pattern in intent["patterns"]]:
            await update.message.reply_text(random.choice(intent["responses"]))
            return

    print("🔎 Информация не найдена в intents.json, обращаемся к GPT...")

    # 🤖 Запрос к GPT
    response = chat_with_gpt(user_input)

    # Проверка на лимит запросов
    if "Превышен лимит запросов" in response:
        await update.message.reply_text(response)
        return

    if response:
        await update.message.reply_text(response)

        # Сохранение нового вопроса и ответа в intents.json
        intents["intents"].append({
            "tag": user_input,
            "patterns": [user_input],
            "responses": [response]
        })
        save_intents(intents)
        return

    # Если ничего не найдено
    await update.message.reply_text("Извините, я не понимаю...")


# 🔹 Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Частые вопросы", "Связаться с ветеринаром"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text("Привет! Я бот Pet. Чем могу помочь?", reply_markup=reply_markup)

# 🔹 Обработчик команды /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🚀 Доступные команды:\n"
        "/start - Перезапустить бота\n"
        "/help - Показать список команд\n"
        "/faq - Частые вопросы\n"
        "/contact - Связаться с ветеринаром\n"
        "Вы также можете написать мне сообщение, и я постараюсь помочь! 😊"
    )
    await update.message.reply_text(help_text)

async def set_bot_commands(application):
    commands = [
        BotCommand("start", "Перезапустить бота"),
        BotCommand("help", "Список команд и помощь"),
        BotCommand("faq", "Частые вопросы"),
        BotCommand("contact", "Связаться с ветеринаром")
    ]
    await application.bot.set_my_commands(commands)

# 🔹 Главная функция запуска бота
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("faq", handle_faq_button))
    app.add_handler(CommandHandler("contact", handle_contact_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
