import jieba
import json
from hanziconv import HanziConv


def cut(str, using_stopword=False, simplified_convert=True):
    if simplified_convert:
        str = HanziConv.toSimplified(str)
    with open('digit_mark.json', encoding='utf-8') as data_file:
        digit_mark = json.load(data_file)
        tokens = [i for i in list(jieba.cut_for_search(str)) if i not in digit_mark]
        if simplified_convert:
            tokens = [HanziConv.toTraditional(i) for i in tokens]
    if using_stopword:
        with open('stopword.json', encoding='utf-8') as data_file:
            stopwords = json.load(data_file)
            tokens = [i for i in list(tokens) if i not in stopwords]
    else:
        tokens = list(tokens)
    return tokens

jieba.load_userdict('userdict.txt')
