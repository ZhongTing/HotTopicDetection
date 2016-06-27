import time

import python_code.feature_extractor as feature_extractor
from python_code.agglomerative_clustering import AgglomerativeClustering
from python_code.clustering_validation import validate_clustering, internal_validate
from python_code.model import ptt_article_fetcher as fetcher
from python_code.model.keywords_extraction import keywords_extraction


def get_ptt_articles(day='NOW/DAY', number=2000):
    return fetcher.fetch_articles('*', number=number, end_day=day, days=1)


def get_cluster_keyword(cluster):
    return [
        keywords_extraction(cluster['articles'], 0),
        keywords_extraction(cluster['articles'], 1)
    ]


def print_clusters(clusters, print_title=False):
    for i in range(len(clusters)):
        cluster = clusters[i]
        score = sum([article.score for article in cluster['articles']])
        print('cluster', i, 'score', score, 'amount', len(cluster['articles']))
        for keywords in get_cluster_keyword(cluster):
            print(keywords)
        if print_title is True:
            for article in cluster['articles']:
                print(article.id, article.title)
            print('\n')


def print_clustering_info(clusters, articles):
    print("\n===============data set information===============")
    print('total articles : ', len(articles))
    print('un-repeat titles : ', len(set([article.title for article in articles])))
    print('total clusters : ', len(clusters))
    print('max_cluster_size : ', max([len(c['articles']) for c in clusters]))


def print_validation_result(labeled_clusters, clusters):
    print("\n===============clustering validation===============")
    validate_result = validate_clustering(labeled_clusters, clusters)
    for key in sorted(validate_result):
        print(key, "{0:.2f}".format(validate_result[key]))


def main(day='NOW/DAY'):
    print('main')
    articles = get_ptt_articles(day, number=1000)
    model = feature_extractor.load_model('model/bin/ngram_300_5_90w.bin')
    t = time.time()
    feature_extractor.ContentRatioExtraction(model, 1, 15, True, 0.6, 0.4).fit(articles)
    clusters = AgglomerativeClustering(0.75).fit(articles)
    tt = time.time()
    print_clustering_info(clusters, articles)
    clusters = sorted(clusters, key=lambda cluster: sum([a.score for a in cluster['articles']]), reverse=True)
    print_clusters(clusters[0:-1], True)
    print('spend', t - tt)
    print('silhouette_index', internal_validate(clusters))


debug_mode = False
if __name__ == '__main__':
    main('2016/06/24')
