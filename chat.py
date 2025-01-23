import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes
from telegram.ext.filters import TEXT

# Load the model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
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

# Function to handle user messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    if user_input.lower() == "quit":
        await update.message.reply_text("Goodbye!")
        return

    # Process user input
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
        await update.message.reply_text("Sorry, I don't understand...")


# Function to start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm Pet. How can I help you today?")


# Main function
def main():
    # Replace 'YOUR_TOKEN_HERE' with your bot's token from BotFather
    app = ApplicationBuilder().token(TOKEN).build()

    # Add command and message handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(TEXT, handle_message))

    # Run the bot
    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
