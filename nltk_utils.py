# nltk_utils.py содержит вспомогательные функции для обработки текста.
# Этот модуль подготавливает текст для обучения нейросети:
# ✔ Разбивает текст (tokenize).
# ✔ Приводит слова к корню (stem).
# ✔ Преобразует в вектор (bag_of_words).

#Импорт библиотек
import nltk # библиотека для обработки естественного языка.
import numpy as np # используется для работы с массивами (в bag_of_words()).

# nltk.download('punkt')
# nltk.download('punkt_tab')

# Инициализация стеммера
# PorterStemmer — инструмент для стемминга (приведения слов к корню): (stemmer.stem("running") → "run", stemmer.stem("flies") → "fli".)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# Функция токенизации
# Разбивает строку на слова: tokenize("Hello, how are you?") → ["Hello", ",", "how", "are", "you", "?"].
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Функция стемминга
# Приводит слово к корню (без приведения к нормальной форме): stem("Running") → "run", stem("Eats") → "eat".
def stem(word):
    return stemmer.stem(word.lower())

# Функция bag_of_words:
# - Преобразует входное предложение в числовой вектор (numpy array).
# - 1.0 означает, что слово есть в предложении, 0.0 — нет.
def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thanks", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,     0 ,      0 ]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag