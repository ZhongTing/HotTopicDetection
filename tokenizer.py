import jieba
import json


def cut(str):
    with open('stopword.json', encoding='utf-8') as data_file:
        stopwords = json.load(data_file)
        tokens = jieba.cut(str)
    tokens = [i for i in list(tokens) if i not in stopwords]
    return tokens
