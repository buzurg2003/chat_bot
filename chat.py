import random
import json
import torch
from collections import Counter
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes
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

# Localization dictionary
translations = {
    "ru": {
        "faq": "Частые вопросы",
        "contact": "Связаться с ветеринаром",
        "start_message": "Привет! Я Pet. Чем могу помочь?",
        "contact_message": "Свяжитесь с ветеринаром, нажав кнопку ниже:",
        "contact_button": "Перейти в чат",
        "unknown": "Извините, я не понимаю...",
        "faq_message": "📌 Часто задаваемые вопросы:\n{}",
        "no_faq": "Пока нет часто задаваемых вопросов."
    },
    "en": {
        "faq": "Frequently Asked Questions",
        "contact": "Contact a Veterinarian",
        "start_message": "Hello! I'm Pet. How can I help you today?",
        "contact_message": "Contact the veterinarian by clicking the button below:",
        "contact_button": "Go to Chat",
        "unknown": "Sorry, I don't understand...",
        "faq_message": "📌 Frequently Asked Questions:\n{}",
        "no_faq": "No frequently asked questions yet."
    }
}

# Request statistics
request_counter = Counter()

# Function to get translation based on user language
def get_translation(update: Update, key: str):
    lang_code = update.effective_user.language_code
    return translations.get(lang_code, translations["en"]).get(key, key)

# Function to handle messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip().lower()  # Normalize user input

    # Get localized button texts
    faq_text = get_translation(update, "faq").strip().lower()
    contact_text = get_translation(update, "contact").strip().lower()

    # Check if the user clicked the buttons
    if user_input == faq_text:
        await handle_faq_button(update, context)
        return
    elif user_input == contact_text:
        await handle_contact_button(update, context)
        return

    # Store request frequency
    request_counter[user_input] += 1

    # Process message with the AI model
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
        await update.message.reply_text(get_translation(update, "unknown"))

# Function to handle "Contact a Veterinarian" button
async def handle_contact_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    username = "buzurg_2003"  # Veterinarian's Telegram username
    keyboard = [[InlineKeyboardButton(get_translation(update, "contact_button"), url=f"tg://resolve?domain={username}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(get_translation(update, "contact_message"), reply_markup=reply_markup)

# Function to handle "Frequently Asked Questions" button
async def handle_faq_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    most_common = request_counter.most_common(5)  # Top 5 most common questions
    if most_common:
        common_text = "\n".join([f"{q[0]} ({q[1]} times)" for q in most_common])
        await update.message.reply_text(get_translation(update, "faq_message").format(common_text))
    else:
        await update.message.reply_text(get_translation(update, "no_faq"))

# Function to start the bot with localized buttons
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[get_translation(update, "faq"), get_translation(update, "contact")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text(get_translation(update, "start_message"), reply_markup=reply_markup)

# Main function
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    # Add handlers for commands and messages
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(TEXT, handle_message))

    # Start bot
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
