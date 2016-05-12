from code.ptt_article_fetcher import fetch_articles, Article
import gensim.models
from gensim import matutils
from code.tokenizer import cut
from numpy import array, dot
import re
from copy import deepcopy
# import random


def get_test_articles():
    all_clustering = [
        # fetch_articles("人生勝利組", number=5),
        # fetch_articles("死刑", number=5),
        fetch_articles("*", number=2000)
    ]
    articles = []
    for clustering in all_clustering:
        articles.extend(clustering)
    # random.shuffle(articles)
    return articles


def get_mock_articles(title_list):
    articles = []
    for title in title_list:
        articles.append(Article({'title': [title]}))
    return articles


def load_model():
    model_path = 'bin/model_5_78w.bin'
    model = gensim.models.Word2Vec.load(model_path)
    print('load word2vec model from ' + model_path)
    return model


def compute_vector(model, string, need_log=False):
    tokens = cut(string, using_stopwords=True, simplified_convert=True, log=need_log)
    if len(tokens) > 0 and (tokens[-1] in ['八卦', '卦']):
        del tokens[-1]
    if need_log is True:
        print(tokens)
    tokens_not_found = [word for word in tokens if word not in model]
    if len(tokens_not_found) is not 0:
        print('token not in model :' + " ".join(tokens_not_found))
    v1 = [model[word] for word in tokens if word in model]
    if len(v1) is 0:
        print('invalid article: \'' + string + '\'')
        return None
    vector = matutils.unitvec(array(v1, float).mean(axis=0))
    return vector


def compute_article_vector(model, articles):
    for article in articles:
        article.vector = compute_vector(model, article.title)


def test_vector_centroid_similarity():

    model = load_model()
    articles = get_test_articles()
    compute_article_vector(model, articles)

    vector_stack = []
    for i in range(0, len(articles)):
        article = articles[i]
        vector = article.vector
        vector_stack.append(vector)
        if len(vector_stack) > 1:
            print(article.title)
            print(articles[i - 1].title)
            si = dot(vector_stack[-1], vector_stack[-2])
            print('previous similarity = ' + str(si))
            # mean
            si2 = dot(array(vector_stack[0:-1]).mean(axis=0), vector_stack[-1])
            print('mean similarity = ' + str(si2))


def test_clustering(articles=None):
    model = load_model()
    if articles is None:
        articles = get_test_articles()
    compute_article_vector(model, articles)
    clusters = []
    for article in articles:
        current_fit_cluster = None
        current_max_similarity = 0

        if article.vector is None:
            print('===============fuck')
            continue
        for cluster in clusters:
            compute_similarity = dot(article.vector, cluster['centroid'])
            try:
                if compute_similarity > current_max_similarity:
                    current_max_similarity = compute_similarity
                    current_fit_cluster = cluster
            except ValueError:
                print(article)
                compute_vector(model, article.title, True)
                print(ValueError)
                print(article.vector)
                return

        if current_max_similarity < threshold:
            log('new cluster ' + article.title)
            if current_fit_cluster is not None:
                log('compare with cluster ' + current_fit_cluster['articles'][0].title)
                log('with max similarity ' + str(current_max_similarity) + "\n")
            clusters.append({'centroid': article.vector, 'articles': [article]})
        else:
            size = len(current_fit_cluster['articles'])
            current_fit_cluster['centroid'] = (
                current_fit_cluster['centroid'] * size + article.vector) / (size + 1)
            current_fit_cluster['articles'].append(deepcopy(article))

    print('-------------------')
    current_max_cluster_size = 0
    min_cluster_count = 0
    min_cluster_size = 2
    # cluster_with_same_title = 0
    for i in range(len(clusters)):
        size = len(clusters[i]['articles'])
        if size > current_max_cluster_size:
            current_max_cluster_size = size
        if size < min_cluster_size:
            min_cluster_count += 1
            continue
        # if len(set([article.title for article in clusters[i]['articles']])) < 2:
        #     print(set([article.title for article in clusters[i]['articles']]))
        #     cluster_with_same_title += 1
        #     continue

        print('cluster ' + str(i))
        for article in clusters[i]['articles']:
            print(article.title)

    print('total articles : ', len(articles))
    print('un-repeat titles : ', len(set([article.title for article in articles])))
    print('total clusters : ', len(clusters))
    # print('clusters with same title ', cluster_with_same_title)
    # print('clusters with different title ', len(clusters) - cluster_with_same_title)
    print('max_cluster_size : ', current_max_cluster_size)
    print('clusters size under {} : {}'.format(min_cluster_size, min_cluster_count))


def log(string):
    if debug_mode is True:
        print(string)


def validate(data):
    model = load_model()
    for i in range(1, len(data)):
        print(data[i - 1])
        vector1 = compute_vector(model, data[i - 1], True)
        print(data[i])
        vector2 = compute_vector(model, data[i], True)
        if vector1 is None or vector2 is None:
            print('vector error')
            continue
        similarity = dot(vector1, vector2)
        print(similarity)


def simulate(file_name, cluster_number):
    with open('word2vec_log/clustering_log/' + file_name, 'r', encoding='utf8') as file:
        content = file.read()
        pattern = re.compile('cluster ' + str(cluster_number) + '\n([\W\w]*?)\ncluster')
        titles = re.findall(pattern, content)[0].split('\n')
        validate(titles)


debug_mode = True
threshold = 0.55
# test_vector_centroid_similarity()
# test_clustering(get_mock_articles(data))
test_clustering()
# simulate('20160509_2000_remove八卦', cluster_number=119)
