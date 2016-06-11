import os
import jieba
import json
from hanziconv import HanziConv


def cut(string, using_stopwords=False, simplified_convert=True, log=False):
    string = string.lower()
    if simplified_convert:
        string = HanziConv.toSimplified(string)
    with open(os.path.join(BASE_DIR, 'digit_mark.json'), encoding='utf-8') as data_file:
        digit_mark = json.load(data_file)
        for digit in digit_mark:
            string = string.replace(digit, ' ')
        tokens = list(jieba.cut_for_search(string))
        if simplified_convert:
            tokens = [HanziConv.toTraditional(i) for i in tokens]
        tokens = [i for i in tokens if i.strip() != '']
    if using_stopwords:
        with open(os.path.join(BASE_DIR, 'stopwords.txt'), encoding='utf-8') as data_file:
            stopwords = [line.replace('\n', '') for line in data_file.readlines()]
            if log:
                removed_tokens = [i for i in list(tokens) if i in stopwords]
                if len(removed_tokens) > 0:
                    print('token removed : ' + ", ".join(removed_tokens))
            tokens = [i for i in list(tokens) if i not in stopwords]
    else:
        tokens = list(tokens)
    return tokens

BASE_DIR = os.path.dirname(__file__)
jieba.load_userdict(os.path.join(BASE_DIR, 'user_dict.txt'))
