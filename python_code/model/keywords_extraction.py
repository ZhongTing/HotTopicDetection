import jieba.analyse
import python_code.model.ptt_article_fetcher as fetcher
import python_code.model.lda as lda
import os


def keywords_extraction(articles, algorithm=0, k=10):
    if not isinstance(articles, list):
        articles = [articles]
    input_data = " ".join([article.title + ' ' + article.content for article in articles])
    if algorithm == 0:
        model = lda.build_lda_model(input_data, 1)
        return lda.get_topic(model, num_topics=1, num_words=k)[0]
    elif algorithm == 1:
        return jieba.analyse.extract_tags(input_data, topK=k, withWeight=False, allowPOS=())
    else:
        return None


def test_keyword_extraction():
    articles = fetcher.fetch_articles('福祿猴', 100)
    print(keywords_extraction(articles, 0))
    print(keywords_extraction(articles, 1))

BASE_DIR = os.path.dirname(__file__)
jieba.analyse.set_stop_words(os.path.join(BASE_DIR, 'my_tokenize/stopwords.txt'))
