import re

from gensim import corpora, models

from python_code.model.ptt_article_fetcher import fetch_articles
from python_code.model.my_tokenize.tokenizer import cut


def build_lda_model(input_data, num_topics=1):
    if len(input_data) == 0:
        print('data is empty')
        return

    if isinstance(input_data, str):
        input_data = [input_data]

    texts = []
    for data in input_data:
        tokens = cut(data, using_stopwords=True, simplified_convert=True)
        texts.append(tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.ldamodel.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=1)

    return lda_model


def get_topic(model, num_topics=1, num_words=15):
    pattern = re.compile('\*([^ ]*)')
    result = {}
    for topic_tuple in model.show_topics(num_topics, num_words):
        # print(topic_tuple)
        words = pattern.findall(topic_tuple[1])
        result[topic_tuple[0]] = words
    return result


def term_expansion(keyword, num_article_for_search=5):
    articles = fetch_articles(keyword, num_article_for_search, days=3)
    if len(articles) == 0:
        print('articles not found...try another search range?')
        return
    input_data = [a.title + " " + a.content for a in articles]
    model = build_lda_model(input_data, 1)
    topic_tokens = get_topic(model)[0]
    return topic_tokens
