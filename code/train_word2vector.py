from code.ptt_article_fetcher import fetch_articles
from code.tokenizer import cut
import os
import time
from gensim.models import Word2Vec


def get_sentence(keyword, number, page=1):
    articles = fetch_articles(keyword, number, page=page, fl='title, content', desc=False)

    result_sentences = []
    for article in articles:
        tokens = cut(article.title, using_stopwords=False, simplified_convert=True)
        result_sentences.append(tokens)
        if hasattr(article, 'content_sentence'):
            for sen in article.content_sentence:
                result_sentences.append(cut(sen, using_stopwords=False, simplified_convert=True))
        if hasattr(article, 'content'):
            result_sentences.append(cut(article.content, using_stopwords=False, simplified_convert=True))
    return result_sentences


title_number = 800000
model_dir = 'bin'
model_path = model_dir + '/whole_content_1_100_' + str(title_number) + '.bin'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

sentences = get_sentence('', title_number)
start_time = time.time()
try:
    model = Word2Vec.load(model_path)
    model.train(sentences)
except FileNotFoundError:
    print('create new word2vec model')
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)

end_time = time.time()
model.save(model_path)
print(model)
print('model_spend_time ' + str(end_time - start_time))
