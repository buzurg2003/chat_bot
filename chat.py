import random
import json
import torch
from collections import Counter
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes
from telegram.ext.filters import TEXT

# Load the model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

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

bot_name = "Pet"
TOKEN = "7999516169:AAEKUaq1we5S9vl4AHYvzazzLTJIx971_Nc"

# Статистика повторяющихся запросов
request_counter = Counter()

# Функция обработки сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.lower()

    if user_input == "quit":
        await update.message.reply_text("Goodbye!")
        return

    # Проверяем, не нажал ли пользователь кнопку "Частые вопросы" или "Связаться с ветеринаром"
    if user_input == "частые вопросы":
        await handle_faq_button(update, context)
        return
    elif user_input == "связаться с ветеринаром":
        await handle_contact_button(update, context)
        return

    # Запоминаем частоту запросов
    request_counter[user_input] += 1

    # Обрабатываем сообщение
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                await update.message.reply_text(response)
                return
    else:
        await update.message.reply_text("Извините, я не понимаю...")

# Функция для вывода контактной информации
async def handle_contact_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = "buzurg_2003"  # Замените на настоящий username
    keyboard = [[InlineKeyboardButton("Перейти в чат", url=f"tg://resolve?domain={username}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Свяжитесь с ветеринаром, нажав кнопку ниже:", reply_markup=reply_markup)

# Функция для вывода часто задаваемых вопросов
async def handle_faq_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    most_common = request_counter.most_common(5)  # Топ-5 популярных запросов
    if most_common:
        common_text = "\n".join([f"{q[0]} ({q[1]} раз)" for q in most_common])
        await update.message.reply_text(f"📌 Часто задаваемые вопросы:\n{common_text}")
    else:
        await update.message.reply_text("Пока нет часто задаваемых вопросов.")

# Функция запуска бота с кнопками в нижнем меню
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Частые вопросы", "Связаться с ветеринаром"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text("Hello! I'm Pet. How can I help you today?", reply_markup=reply_markup)

# Главная функция
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    # Добавляем обработчики команд и сообщений
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(TEXT, handle_message))

    # Запускаем бота
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
