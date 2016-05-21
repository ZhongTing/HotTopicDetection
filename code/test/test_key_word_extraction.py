import jieba.analyse
import code.model.ptt_article_fetcher as fetcher
import code.model.lda as lda


def test_keyword_extraction():
    articles = fetcher.fetch_articles('福祿猴', 100)

    input_data = " ".join([article.title + ' ' + article.content for article in articles])

    result = jieba.analyse.extract_tags(input_data, topK=10, withWeight=False, allowPOS=())
    print(result)
    model = lda.build_lda_model(input_data, 1)
    result = lda.get_topic(model, 1, 10)
    print(result[0])


test_keyword_extraction()
