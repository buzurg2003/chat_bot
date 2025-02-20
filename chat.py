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

# Глобальная переменная для хранения времени сброса лимита
gpt_reset_time = None


def load_intents():
    with open("intents.json", "r", encoding="utf-8") as file:
        return json.load(file)

def get_intent_response(user_input, intents):
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if user_input.lower() in pattern.lower():
                return intent["responses"][0]
    return None

api_key="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"

import time
import openai


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
        error_data = getattr(e, "body", {})
        error_code = error_data.get("error", {}).get("code", "")

        if error_code == "insufficient_quota":
            return "Превышен лимит запросов. Пополните баланс или проверьте подписку OpenAI."

        reset_time = error_data.get("x-ratelimit-reset")
        if reset_time:
            wait_time = int(reset_time) - int(time.time())
            if wait_time > 0:
                return f"Превышен лимит запросов. Попробуйте снова через {wait_time} секунд."

        return "Превышен лимит запросов. Попробуйте позже или проверьте подписку OpenAI."


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words("russian")]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    pos_tags = nltk.pos_tag(tokens)
    return [word for word, tag in pos_tags]

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

request_counter = Counter()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.lower()
    intent_response = get_intent_response(user_input, intents)  # Добавил intents

    if intent_response:
        await update.message.reply_text(intent_response)
        return  # Если нашли ответ в intents.json, не идем дальше

    print("🔎 Информация не найдена в intents.json, обращаемся к GPT...")

    response = chat_with_gpt(user_input)

    if "Превышен лимит запросов" in response:
        await update.message.reply_text(response)
        return  # Если лимит превышен, сразу выходим

    await update.message.reply_text(response)

    if user_input == "quit":
        await update.message.reply_text("Goodbye!")
        return

    if user_input == "частые вопросы":
        await handle_faq_button(update, context)
        return
    elif user_input == "связаться с ветеринаром":
        await handle_contact_button(update, context)
        return

    request_counter[user_input] += 1

    normalized_text = preprocess_text(user_input)
    X = bag_of_words(normalized_text, all_words)
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


async def handle_contact_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = "buzurg_2003"
    keyboard = [[InlineKeyboardButton("Перейти в чат", url=f"tg://resolve?domain={username}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Свяжитесь с ветеринаром, нажав кнопку ниже:", reply_markup=reply_markup)

async def handle_faq_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    most_common = request_counter.most_common(5)
    if most_common:
        common_text = "\n".join([f"{q[0]} ({q[1]} раз)" for q in most_common])
        await update.message.reply_text(f"📌 Часто задаваемые вопросы:\n{common_text}")
    else:
        await update.message.reply_text("Пока нет часто задаваемых вопросов.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Частые вопросы", "Связаться с ветеринаром"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text("Hello! I'm Pet. How can I help you today?", reply_markup=reply_markup)

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(TEXT, handle_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()

# ! Google AI API: https://aistudio.google.com/apikey
# ! Open AI: https://platform.openai.com/usage
# ! Platform Open AI: https://platform.openai.com/docs/overview
