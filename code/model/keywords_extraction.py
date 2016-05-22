import jieba.analyse
import code.model.ptt_article_fetcher as fetcher
import code.model.lda as lda


def keywords_extraction(articles, algorithm):
    input_datas = " ".join([article.title + ' ' + article.content for article in articles])
    if algorithm == 0:
        model = lda.build_lda_model(input_datas, 1)
        return lda.get_topic(model, num_topics=1, num_words=5)[0]
    elif algorithm == 1:
        return jieba.analyse.extract_tags(input_datas, topK=10, withWeight=False, allowPOS=())
    else:
        return None


def test_keyword_extraction():
    articles = fetcher.fetch_articles('福祿猴', 100)
    print(keywords_extraction(articles, 0))
    print(keywords_extraction(articles, 1))
