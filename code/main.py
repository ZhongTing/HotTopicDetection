import re
from copy import deepcopy
import gensim.models
from gensim import matutils
from numpy import array, dot
from code.model.ptt_article_fetcher import Article
from code.model.my_tokenize.tokenizer import cut
from code.test.make_test_data import get_test_clusters
import code.model.lda as lda
import random
from code.clustering_validation import validate_clustering
import time


def get_test_articles(clusters=get_test_clusters()):
    articles = []
    for cluster in clusters:
        articles.extend(cluster['articles'])
    random.shuffle(articles)
    return articles


def get_mock_articles(title_list):
    articles = []
    for title in title_list:
        articles.append(Article({'title': [title]}))
    return articles


def load_model():
    t = time.time()
    model_path = 'model/bin/model_82w.bin'
    model = gensim.models.Word2Vec.load(model_path)
    t = int(time.time() - t)
    print('spend {}s to load word2vec model from {}'.format(t, model_path))
    return model


def compute_vector(model, string, need_log=False):
    tokens = cut(string, using_stopwords=True, simplified_convert=True, log=need_log)
    if len(tokens) > 0 and (tokens[-1] in ['八卦', '卦']):
        del tokens[-1]
    if need_log is True:
        print(tokens)
    tokens_not_found = [word for word in tokens if word not in model]
    if len(tokens_not_found) is not 0:
        log('token not in model :' + " ".join(tokens_not_found))
    v1 = [model[word] for word in tokens if word in model]
    if len(v1) is 0:
        print('invalid article: \'' + string + '\'')
        return None
    vector = matutils.unitvec(array(v1, float).mean(axis=0))
    return vector


def compute_article_vector(model, articles):
    for article in articles:
        article.vector = compute_vector(model, article.title)


def initialize_clusters(articles):
    clusters = []
    for article in articles:
        if article.vector is None:
            print('initial error, invalid article: \'' + article.title + '\'')
            continue

        not_found = True
        for cluster in clusters:
            if cluster['articles'][-1].title == article.title:
                cluster['articles'].append(article)
                not_found = False
                break

        if not_found:
            clusters.append({'centroid': article.vector, 'articles': [article]})
    return clusters


def get_cluster_keyword(cluster):
    input_datas = [a.title + ' ' + a.content for a in cluster['articles']]
    model = lda.build_lda_model(input_datas, 1)
    return lda.get_topic(model, num_topics=1, num_words=5)[0]


def merge_clusters(clusters):
    clusters_after_merge = []
    while len(clusters) != 0:
        target_cluster = clusters.pop()
        highest_similarity = 0
        candidate_cluster = None
        for cluster in clusters:
            similarity = compute_similarily(cluster, target_cluster)
            if similarity > highest_similarity:
                highest_similarity = similarity
                candidate_cluster = cluster
        if highest_similarity > threshold:
            log('merged: cluster {} with {}'.format(
                candidate_cluster['articles'][0].title, target_cluster['articles'][0].title))
            combined(candidate_cluster, target_cluster)
        else:
            clusters_after_merge.append(target_cluster)
    return clusters_after_merge


def combined(cluster, merged_cluster):
    cluster['articles'].extend(merged_cluster['articles'])
    cluster['centroid'] = sum([a.vector for a in cluster['articles']]) / len(cluster['articles'])


def compute_similarily(cluster_a, cluster_b):
    return dot(cluster_a['centroid'], cluster_b['centroid'])


def print_clusters(clusters):
    for i in range(len(clusters)):
        print('cluster ', i)
        print(get_cluster_keyword(clusters[i]))


def print_clustering_result(labeled_clusters, clusters, articles):
    print("\n===============data set information===============")
    print('total articles : ', len(articles))
    print('un-repeat titles : ', len(set([article.title for article in articles])))
    print('total clusters : ', len(clusters))
    print('max_cluster_size : ', max([len(c['articles']) for c in clusters]))

    print("\n===============clustering validation===============")
    validate_result = validate_clustering(labeled_clusters, clusters)
    for key in sorted(validate_result):
        print(key, "{0:.2f}".format(validate_result[key]))


def test_clustering2():
    model = load_model()
    labeled_clusters = get_test_clusters()
    articles = get_test_articles(labeled_clusters)
    compute_article_vector(model, articles)
    clusters = initialize_clusters(articles)
    clusters = merge_clusters(clusters)
    # print_clusters(clusters)
    print_clustering_result(labeled_clusters, clusters, articles)


def test_clustering():
    model = load_model()
    labeled_clusters = get_test_clusters()
    articles = get_test_articles(labeled_clusters)
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

    print_clustering_result(labeled_clusters, clusters, articles)


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


debug_mode = False
threshold = 0.5
test_clustering()
# simulate('20160509_2000_remove八卦', cluster_number=119)
