import os
import time
from datetime import datetime

from gensim.models import Word2Vec
from code.model.ptt_article_fetcher import fetch_articles
from code.model.my_tokenize.tokenizer import cut


def get_sentence(keyword, number, page=1):
    articles = fetch_articles(keyword, number, page=page, fl='title, content', desc=False)

    result_sentences = []
    for article in articles:
        tokens = cut(article.title, using_stopwords=False, simplified_convert=True)
        result_sentences.append(tokens)
        # if hasattr(article, 'content_sentence'):
            # for sen in article.content_sentence:
                # result_sentences.append(cut(sen, using_stopwords=False, simplified_convert=True))
        if hasattr(article, 'content'):
            result_sentences.append(cut(article.content, using_stopwords=False, simplified_convert=True))
    return result_sentences


def train(model_name, article_number):
    model_dir = 'bin'
    model_path = os.path.join(model_dir, model_name + '.bin')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    sentences = get_sentence('', article_number)
    start_time = time.time()
    try:
        model = Word2Vec.load(model_path)
        model.train(sentences)
    except FileNotFoundError:
        print('create new word2vec model')
        model = Word2Vec(size=300, window=5, min_count=10, workers=8, sg=1)
        model.build_vocab(sentences)
        print('build vocab spend ', time.time() - start_time)
        model.train(sentences)

    end_time = time.time()
    model.save(model_path)
    print(model)
    print('model_spend_time ' + str(end_time - start_time))

print(datetime.now())
train('ngram_300_10_85w_sentence_one_word', 850000)
print(datetime.now())
