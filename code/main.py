from ptt_article_fetcher import fetch_articles
import gensim.models
from gensim import matutils
from tokenizer import cut
from numpy import array, dot


def main():
    model_path = 'bin/model_78w.bin'
    model = gensim.models.Word2Vec.load(model_path)
    print('load word2vec model from ' + model_path)
    all_clusting = [
        fetch_articles("人生勝利組", number=10),
        fetch_articles("死刑", number=10)
    ]
    articles = []
    for clusting in all_clusting:
        articles.extend(clusting)

    for article in articles:
        article.tokens = cut(article.title, using_stopword=False, simplified_convert=True)
        v1 = [model[word] for word in article.tokens if word in model]
        article.vector = matutils.unitvec(array(v1, float).mean(axis=0))

    vector_stack = []
    for i in range(0, len(articles)):
        article = articles[i]
        print(article.title)
        vector = article.vector
        vector_stack.append(vector)

        if len(vector_stack) > 1:
            si = dot(vector_stack[-1], vector_stack[-2])
            print('previos similarity = ' + str(si))
            # mean
            si2 = dot(array(vector_stack[0:-1]).mean(axis=0), vector_stack[-1])
            print('mean similarity = ' + str(si2))


main()
