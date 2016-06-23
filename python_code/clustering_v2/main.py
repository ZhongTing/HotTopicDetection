import time

from python_code.clustering_v2.feature_extractor import FeatureExtractor

from python_code.clustering_v2.agglomerative_clustering import AgglomerativeClustering
from python_code.clustering_validation import validate_clustering
from python_code.model import ptt_article_fetcher as fetcher
from python_code.model.keywords_extraction import keywords_extraction


def get_ptt_articles(number=2000):
    return fetcher.fetch_articles('*', number=number, days=1)


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


def main(threshold=0.55):
    print('main', threshold)
    articles = get_ptt_articles(number=2000)

    FeatureExtractor('model/bin/ngram_300_3_83w.bin').fit_with_extraction_ratio(articles, 1, 15, 0.4, 0.6)
    t = time.time()
    clusters = AgglomerativeClustering(0.65).fit(articles)
    tt = time.time()
    print_clustering_info(clusters, articles)
    clusters = sorted(clusters, key=lambda cluster: sum([a.score for a in cluster['articles']]), reverse=True)
    print_clusters(clusters[0:5], True)
    print('spend', t - tt)
    # print('silhouette_index', internal_validate(clusters))


debug_mode = False
if __name__ == '__main__':
    main(4)
