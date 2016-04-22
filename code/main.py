import sys
from tokenizer import cut
from ptt_article_fetcher import fetch_articles
import re
import time
import os
from gensim import corpora, models


def log(file, object):
    print(object)
    print(object, end="\n", file=file)


def build_lda_by_keyword(keywords, num_article_for_search, num_topics=0):
    if num_topics == 0:
        num_topics = len(keywords)

    dir_name = '../data'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    filename = dir_name + '/' + str("_".join(keywords) + "_" + str(num_article_for_search))

    file = open(filename, 'w', encoding="utf-8")
    articles = []
    for keyoword in keywords:
        articles_keyword = fetch_articles(keyoword, num_article_for_search, days=3)
        articles += articles_keyword
        log(file, "%s : %d" % (keyoword, len(articles_keyword)))

    texts = []
    for article in articles:
        tokens = cut(article.title + article.content)
        texts.append(tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    start = time.time()
    ldamodel = models.ldamodel.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=1)

    pattern = re.compile('\*(.*?) ')
    for topic in ldamodel.show_topics(num_topics=num_topics, num_words=15):
        log(file, pattern.findall(topic[1]))
    end = time.time()
    log(file, "model train time : " + str(end - start))

    print("\n\n\n\n", file=file)
    for article in articles:
        print(article.title, end="\n", file=file)

    file.close()


def test1():
    keywords_candidate = ['隨機', '巴拿馬', '慈濟']
    key_index_set = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    for num_topics in [10, 20, 50, 100]:
        for key_index in key_index_set:
            keywords = []
            for index in key_index:
                keywords.append(keywords_candidate[index])
            build_lda_by_keyword(keywords, num_topics)


def test2():
    keywords = ['王建民', '柯文哲', '和田光司', '義美', '統神']
    build_lda_by_keyword(keywords, 9)


args = sys.argv

if len(args) > 2:
    build_lda_by_keyword(args[2:], args[1])
else:
    build_lda_by_keyword(["日本"], 500, 1)
