import string
from collections import Counter
import torch

from nltk.corpus import webtext, stopwords
import nltk
from natasha import MorphVocab, Doc, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Segmenter, NewsNERTagger
import pymorphy2
from pymystem3 import Mystem

def tokenization():
    # Загрузить коллекцию webtext
    nltk.download('webtext')

    # Загрузить текст 'grail.txt'
    data = webtext.raw('grail.txt')

    # Заменить все переносы строк на пробелы
    data = data.replace('\n', ' ')

    # Токенизировать текст по пробелам
    tokens = data.split()
    tokens = [token for token in tokens if token]

    # Посчитать количество слов без пустого слова
    word_count = len(tokens)

    print(f"Word count {word_count}")

    # Удалить пустые токены и посчитать количество уникальных слов
    unique_tokens = set(tokens) - {''}
    unique_word_count = len(unique_tokens)

    print(f"Unique word count {unique_word_count}")

    # Посчитать частоту встречаемости каждого уникального слова
    word_freq = Counter(tokens)

    # Найти самое частое непустое слово
    most_common_word = word_freq.most_common(1)

    print(f"Most common word {most_common_word}")


def filtering():
    nltk.download('webtext')
    nltk.download('stopwords')

    # Загрузить текст 'grail.txt'
    data = webtext.raw('grail.txt')

    # Заменить все переносы строк на пробелы
    data = data.replace('\n', ' ')

    # Удалить все символы пунктуации
    data = data.translate(str.maketrans('', '', string.punctuation))

    # Токенизировать текст по пробелам и привести к нижнему регистру
    tokens = data.lower().split()
    tokens = [token for token in tokens if token]

    # Удалить все слова, содержащиеся в списке стоп-слов
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Посчитать частоту встречаемости каждого уникального слова
    word_freq = Counter(filtered_tokens)

    # Найти самое частое непустое слово
    most_common_word = word_freq.most_common(5)

    print(most_common_word)


def lemmatization():
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)

    text1 = 'вкусное жаркое'
    text2 = 'жаркое лето'
    doc1 = Doc(text1)
    doc2 = Doc(text2)

    def lemmatize(doc):
        doc.segment(Segmenter())
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)

        sent = doc.sents[0]
        sent.morph.print()

        morph_vocab = MorphVocab()

        for token in doc.tokens:
            print(token)
            token.lemmatize(morph_vocab)
            print(token.lemma)

    lemmatize(doc1)
    lemmatize(doc2)


def pymorphy():
    morph = pymorphy2.MorphAnalyzer()

    text = "вкусное жаркое"
    text2 = "жаркое лето"

    def lemmatize(text):
        words = text.split()
        res = list()
        for word in words:
            p = morph.parse(word)[0]
            res.append(p.normal_form)
            res.append(p.tag)

        return res

    print(lemmatize(text))
    print(lemmatize(text2))


def mystem():
    mystem = Mystem()

    sentence = "вкусное жаркое"
    sentence2 = "жаркое лето"

    def lemmatize(text):
        lemmas = mystem.lemmatize(text)

        lemmatized_sentence = ''.join(lemmas).strip()

        print(lemmatized_sentence)

    lemmatize(sentence)
    lemmatize(sentence2)


def tensor():
    # Заданный тензор x
    x = torch.tensor([[4.0, 6.0], [8.0, 5.0]], requires_grad=True)

    # 1. Возведение в квадрат
    y = x ** 2
    print(y)

    # 2. Подстановка в выражение (y-2)(y+2)
    z = y ** 2 - 4
    print(z)

    # 3. Вычисление среднего значения
    mean_z = torch.mean(z)
    print(f'Mean {mean_z}')

    # Вызов функции backward для среднего значения
    mean_z.backward()
    print(mean_z)

    # Сумма всех элементов матрицы градиентов для тензора x
    grad_sum = x.grad.sum().item()

    print("Сумма всех элементов матрицы градиентов для тензора x:", grad_sum)

# pymorphy()
# tokenization()
# filtering()
# lemmatization()
# mystem()
tensor()
