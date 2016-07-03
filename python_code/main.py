import os
import time

import python_code.feature_extractor as feature_extractor
from python_code.agglomerative_clustering import AgglomerativeClustering
from python_code.clustering_validation import validate_clustering, internal_validate
from python_code.model import ptt_article_fetcher as fetcher
from python_code.model.keywords_extraction import keywords_extraction


def get_ptt_articles(day='NOW/DAY', number=2000):
    return fetcher.fetch_articles('*', number=number, end_day=day, days=0)


def get_cluster_keyword(cluster):
    return [
        keywords_extraction(cluster['articles'], 0),
        keywords_extraction(cluster['articles'], 1)
    ]


def print_clusters(clusters, print_title=False, file=None):
    for i in range(len(clusters)):
        cluster = clusters[i]
        score = sum([article.score for article in cluster['articles']])
        print('cluster', i, 'score', score, 'amount', len(cluster['articles']), file=file)
        for keywords in get_cluster_keyword(cluster):
            print(keywords, file=file)
        if print_title is True:
            for article in cluster['articles']:
                # print(article.id, article.title, file=file)
                print(article.title, file=file)
            print('\n', file=file)


def print_clustering_info(clusters, articles, file):
    print("\n===============data set information===============", file=file)
    print('total articles : ', len(articles), file=file)
    print('un-repeat titles : ', len(set([article.title for article in articles])), file=file)
    print('total clusters : ', len(clusters), file=file)
    print('max_cluster_size : ', max([len(c['articles']) for c in clusters]), file=file)


def print_validation_result(labeled_clusters, clusters):
    print("\n===============clustering validation===============")
    validate_result = validate_clustering(labeled_clusters, clusters)
    for key in sorted(validate_result):
        print(key, "{0:.2f}".format(validate_result[key]))


def main(day='NOW/DAY', log=False):
    print('main', day)
    file = None
    if log is True:
        path = '../log/TopFiveTopic'
        check_dir(path)
        file = open(os.path.join(path, day.replace('/', '') + '.txt'), 'w', encoding='utf8')
    articles = get_ptt_articles(day, number=5000)
    model = feature_extractor.load_model('model/bin/ngram_300_5_90w.bin')
    t = time.time()
    feature_extractor.ContentRatioExtraction(model, 1, 15, True, 0.5, 0.5).fit(articles)
    clusters = AgglomerativeClustering(0.75).fit(articles)
    tt = time.time()
    print_clustering_info(clusters, articles, file)
    clusters = sorted(clusters, key=lambda cluster: sum([a.score for a in cluster['articles']]), reverse=True)
    print_clusters(clusters[0:5], True, file)
    print('spend', t - tt, file=file)
    # print('silhouette_index', internal_validate(clusters))


def log_hot_topic(month):
    for day_counter in range(1, 31):
        day = month + '/' + str(day_counter).zfill(2)
        try:
            main(day, True)
        finally:
            pass


def check_dir(path):
    base_dir = os.path.dirname(__file__)
    dir_path = os.path.join(base_dir, path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


debug_mode = False
if __name__ == '__main__':
    # main('2016/06/24')
    log_hot_topic('2016/06')
