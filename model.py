# model.py - содержит архитектуру нейросети для чат-бота.
# Эта модель — обычная многослойная нейросеть (MLP), которая берёт вектор слов (bag_of_words()) и предсказывает tag.
import torch # torch — основной модуль PyTorch.
import torch.nn as nn # torch.nn — содержит классы для построения нейросетей.

# Создание класса нейросети
# NeuralNet — наследует torch.nn.Module, что делает её стандартной PyTorch-моделью.
class NeuralNet(nn.Module):
    # Инициализация слоёв
    # input_size — размер входного вектора (длина мешка слов).
    # hidden_size — количество нейронов в скрытых слоях.
    # num_classes — количество классов (тегов).
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # первый полносвязный слой.
        self.l2 = nn.Linear(hidden_size, hidden_size) # второй полносвязный слой.
        self.l3 = nn.Linear(hidden_size, num_classes) # выходной слой.
        self.relu = nn.ReLU() # функция активации ReLU (ускоряет обучение).

    # Прямой проход (forward pass)
    def forward(self, x):
        # Данные проходят через 3 полносвязных слоя (l1 → l2 → l3).
        # После первых двух слоёв применяется ReLU(), чтобы избежать линейности.
        # В конце нет softmax(), потому что PyTorch уже включает его в CrossEntropyLoss() (в train.py).
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax

        return out
