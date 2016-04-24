import ptt_article_fetcher
import tokenizer
import os
import time
from gensim.models import Word2Vec


def get_sentence(keyword, number, page=1):
    articles = ptt_article_fetcher.fetch_articles(keyword, number, page=page, only_title=True)

    sentences = []
    for article in articles:
        tokens = tokenizer.cut(article.title, using_stopword=False, simplified_convert=True)
        # print(tokens)
        sentences.append(tokens)

    return sentences

title_number = 1000
model_dir = 'bin'
model_path = model_dir + '/word2vecmodel_' + str(title_number) + '.bin'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

sentences = get_sentence('', title_number)
start_time = time.time()
try:
    model = Word2Vec.load(model_path)
    model.train(sentences)
except FileNotFoundError:
    print('creat new word2vec model')
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

end_time = time.time()
model.save(model_path)
print('model_spend_time ' + str(end_time - start_time))
