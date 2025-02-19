# train.py обучает нейросетевую модель для чат-бота.
# Этот код загружает intents.json, обрабатывает данные, обучает нейросетевую модель и сохраняет её в data.pth.
# Затем этот файл загружается в chat.py для работы чат-бота.

# Импорты библиотек
import json # json — для загрузки intents.json, где хранятся обучающие данные.
# ! nltk_utils — модуль с функциями для обработки текста:
#   tokenize() — разбивает текст на слова,
#   stem() — приводит слова к корню,
#   bag_of_words() — преобразует текст в числовой вектор (мешок слов).
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np # numpy — для работы с массивами.
# torch и torch.nn — фреймворк для создания нейросетей.
import torch
import torch.nn as nn
# Dataset, DataLoader — классы PyTorch для работы с данными.
from torch.utils.data import Dataset, DataLoader
# NeuralNet — твоя нейросеть (из model.py), которая будет обучаться.
from model import NeuralNet

# ! Загрузка intents.json
#   Загружает JSON-файл с обучающими фразами и ответами.
with open('intents.json', 'r', encoding="utf-8") as f:
    intents = json.load(f)

# Обработка данных
all_words = [] # all_words — список всех слов из intents.json.
tags = [] # tags — все категории (tag), которые бот должен распознавать.
xy = [] # xy — список пар (список слов, соответствующий tag).
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# ! Очистка и подготовка словаря
ignore_words = ['?', '!', ".", ","] # Удаляем символы ? ! . ,.
all_words = [stem(w) for w in all_words if w not in ignore_words] # Приводим слова к корневой форме (stem).
all_words = sorted(list(set(all_words))) # Убираем дубликаты и сортируем списки.
tags = sorted(list(set(tags)))

# ! Создание обучающей выборки
X_train = [] # X_train — список мешков слов для каждой фразы.
y_train = [] # y_train — индексы tag, которые соответствуют X_train.
for (pattern_sentence, tag) in xy:
    # Используется bag_of_words(), чтобы представить каждую фразу как вектор.
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)

# ! Создание кастомного датасета
#   Создаёт Dataset, который используется DataLoader-ом для пакетной загрузки данных.
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# ! Hyperparameters (Гиперпараметры)
#   Определяются параметры модели:
batch_size = 8 # размер пакета данных для обучения.
hidden_size = 8 # количество нейронов в скрытом слое.
output_size = len(tags) # количество классов.
input_size = len(X_train[0]) # размерность входных данных.
learning_rate = 0.001 # скорость обучения.
num_epochs = 1000 # число эпох (проходов по данным).

# ! Создание загрузчика данных
#   DataLoader перемешивает (shuffle=True) и загружает данные пакетами (batch_size=8).
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# ! Создание и настройка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Если доступен CUDA, модель будет обучаться на GPU.
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss() # функция потерь для задачи классификации.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # оптимизатор для обновления весов нейросети.

# Цикл обучения
for epoch in range(num_epochs):
    for (words, labels) in train_loader: # words и labels передаются на cuda (если доступно).
        words = words.to(device)
        labels = labels.to(device).long()

        # forward
        outputs = model(words) # Проход вперёд.
        loss = criterion(outputs, labels) # Вычисление ошибки

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward() # Обратное распространение
        optimizer.step() # Обновление весов

    #Каждые 100 эпох выводится текущее значение ошибки.
    if (epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss: {loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

#Сохранение обученной модели
data = {
    "model_state": model.state_dict(), # веса нейросети.
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE) # сохраняет всё в data.pth, который используется chat.py.

print(f'training complete. file saved to {FILE}') # Выводит сообщение о завершении