import random
import re
from collections import OrderedDict

import python_code.test.make_test_data as test_data

# noinspection PyPep8Naming
from python_code.agglomerative_clustering import AgglomerativeClustering as HAC

from numpy import mean, std
import time
import os
import csv

from python_code.clustering_validation import validate_clustering
from python_code.feature_extractor import FeatureExtractor
from python_code.tf_idf_feature_extractor import TFIDFFeatureExtractor

FEATURE_TF = 'tf whole article'
FEATURE_TF_IDF = 'tfidf whole article'
FEATURE_ARTICLE = 'whole article'
FEATURE_ARTICLE_EXTRACTION = 'article with keyword extraction'
FEATURE_TITLE = 'title'
FEATURE_TITLE_EXTRACTION = 'title with keyword extraction'


class AgglomerativeClusteringTester:
    def __init__(self, feature, model_path='model/bin/ngram_300_3_83w.bin', number_article_per_test_cluster=None,
                 use_idf=False):
        if feature != FEATURE_TF_IDF and feature != FEATURE_TF:
            self._feature_extractor = FeatureExtractor(model_path=model_path)
        self._all_test_clusters = test_data.get_test_clusters()
        self._feature_mode = feature
        self._number_article_per_test_cluster = number_article_per_test_cluster
        self.use_idf = use_idf

        if self._feature_mode == FEATURE_TF_IDF or self._feature_mode == FEATURE_TF or self.use_idf:
            pass
        else:
            if use_idf:
                articles = [a for cluster in self._all_test_clusters for a in cluster['articles']]
                self._feature_extraction(self._feature_mode, articles, self.use_idf)
            else:
                for cluster in self._all_test_clusters:
                    self._feature_extraction(self._feature_mode, cluster['articles'], self.use_idf)
        print('tester init with feature', self._feature_mode)

    def _feature_extraction(self, feature, articles, use_idf):
        if feature == FEATURE_TF:
            TFIDFFeatureExtractor(use_idf=False).fit(articles)
        elif feature == FEATURE_TF_IDF:
            TFIDFFeatureExtractor(use_idf=True).fit(articles)
        elif feature == FEATURE_ARTICLE:
            self._feature_extractor.fit(articles, use_idf=use_idf)
        elif feature == FEATURE_ARTICLE_EXTRACTION:
            self._feature_extractor.fit_with_extraction(articles, 1, 15, use_idf=False, with_weight=True)
        elif self._feature_mode == FEATURE_TITLE:
            self._feature_extractor.fit_with_extraction_ratio(articles, t=1, c=0)
        elif self._feature_mode == FEATURE_TITLE_EXTRACTION:
            pass
        else:
            raise ValueError('Feature invalid')

    def _get_test_articles(self, sampling=True):
        articles = []
        self._labeled_clusters = []
        for cluster in self._all_test_clusters:
            cluster_articles = cluster['articles']
            if sampling:
                l = len(cluster_articles)
                if self._number_article_per_test_cluster is not None:
                    if l > self._number_article_per_test_cluster:
                        l = self._number_article_per_test_cluster
                cluster_articles = random.sample(cluster_articles, k=random.randint(1, l))
            self._labeled_clusters.append({'articles': cluster_articles})
            articles.extend(cluster_articles)
        random.shuffle(articles)
        if self._feature_mode == FEATURE_TF or self._feature_mode == FEATURE_TF_IDF or self.use_idf:
            self._feature_extraction(self._feature_mode, articles, self.use_idf)
        return articles

    def stable_test(self, times=3):
        file_name = 'stable_test times={}'.format(times)
        result_table = {}

        for time_counter in range(times):
            articles = self._get_test_articles(False)
            random.shuffle(articles)
            print('time counter', time_counter)
            for key in [HAC(0.55).quick_fit, HAC(0.55).fit]:
                clusters = key(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                algorithm_name = str(key).split(' ')[2]
                if algorithm_name not in result_table:
                    result_table[algorithm_name] = []
                print(result)
                result_table[algorithm_name].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, self._feature_mode, file_name)

    def find_best_threshold(self, linkage, sim, quick, start_th=0.3, end_th=0.8, step=0.05, sampling=True, times=1):
        file_name = 'threshold {} {} quick={} idf={} sampling={} times={}'.format(linkage, sim, quick, self.use_idf,
                                                                                  sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            threshold = start_th
            while threshold < end_th + step:
                print('threshold', threshold)
                if quick is True:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).quick_fit(articles)
                else:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).fit(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                key = '{0:.2f}'.format(threshold)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
                threshold += step
        self._print_test_result(result_table)
        self._save_as_csv(result_table, self._feature_mode, file_name)

    def find_ratio_threshold(self, method, k, t, c, start_th=0.3, end_th=0.8, step=0.05, sampling=True, times=1):
        file_name = 'ratio th method{} k={} t={} c={} sampling={} times={}'.format(method, k, t, c, sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            self._feature_extractor.fit_with_extraction_ratio(articles, method, k, t, c)
            threshold = start_th
            while threshold < end_th + step:
                print('threshold', threshold)
                clusters = HAC(threshold, linkage=HAC.LINKAGE_CENTROID, similarity=HAC.SIMILARITY_DOT).quick_fit(
                    articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                key = 'th{} method{} k{} t{} c{}'.format(threshold, method, k, t, c)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
                threshold += step
        self._print_test_result(result_table)
        self._save_as_csv(result_table, self._feature_mode, file_name)

    def compare_ratio(self, method, k, args, sampling=True, times=1):
        file_name = 'compare ratio method{} k={} sampling={} times={}'.format(method, k, sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            for t, c, threshold in args:
                print('t ratio', t)
                self._feature_extractor.fit_with_extraction_ratio(articles, method, k, t, c)
                clusters = HAC(threshold, linkage=HAC.LINKAGE_CENTROID, similarity=HAC.SIMILARITY_DOT).fit(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                key = 't{} c{} th{} method{} k{} '.format(t, c, threshold, method, k)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, self._feature_mode, file_name)

    def compare(self, sim, quick, args, sampling=False, times=1):
        file_name = 'compare {} quick={} sampling={} times={}'.format(sim, quick, sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            for linkage, threshold in args:
                if quick is True:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).quick_fit(articles)
                else:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).fit(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                key = '{}-{}'.format(linkage, threshold)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)

        self._print_test_result(result_table)
        self._save_as_csv(result_table, self._feature_mode, file_name)

    def compare_time_feature(self, name, threshold, linkage, sim, sampling=False, times=1):
        file_name = '{} {} sampling={} times={}'.format(name, self._number_article_per_test_cluster, sampling,
                                                        times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            for i in range(3):
                t = time.time()
                if i == 0:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).fit(articles)
                    key = 'normal {} {} {}'.format(linkage, threshold, sim)
                elif i == 1:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).quick_fit(articles, time_order=True)
                    key = 'time_order {} {} {}'.format(linkage, threshold, sim)
                else:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).quick_fit(articles, time_order=False)
                    key = 'random {} {} {}'.format(linkage, threshold, sim)

                result = validate_clustering(self._labeled_clusters, clusters)
                result['time'] = time.time() - t
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, 'compare all', file_name)

    def compare_different_method(self, name, args, sampling=False, times=1):
        file_name = '{} {} sampling={} times={}'.format(name, self._number_article_per_test_cluster, sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            for (feature, linkage, threshold, sim, quick, use_idf) in args:
                t = time.time()
                self._feature_extraction(feature, articles, use_idf=use_idf)
                if quick:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).quick_fit(articles)
                else:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).fit(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                result['time'] = time.time() - t
                key = '{} {} {} {} {} {}'.format(feature, linkage, threshold, sim, quick, use_idf)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, 'compare all', file_name)

    def compare_extraction(self, args, sampling=False, times=1):
        file_name = 'extraction {} sampling={} times={}'.format(self._number_article_per_test_cluster, sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            for (method, k, linkage, threshold, with_weight) in args:
                t = time.time()
                invalid_id_list = self._feature_extractor.fit_with_extraction(articles, method, k,
                                                                              with_weight=with_weight)
                for invalid_id in invalid_id_list:
                    removed = False
                    for cluster in self._labeled_clusters:
                        if removed:
                            break
                        for article in cluster['articles']:
                            if article.id == invalid_id:
                                cluster['articles'].remove(article)
                                removed = True
                                break

                clusters = HAC(threshold, linkage=linkage, similarity=HAC.SIMILARITY_DOT).quick_fit(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                result['time'] = time.time() - t
                key = 'method{} k={} {} {} weight={}'.format(method, k, linkage, threshold, with_weight)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, 'compare all', file_name)

    @staticmethod
    def _print_test_result(result_table):
        first_key = list(result_table.keys())[0]
        for key in sorted(result_table[first_key][0].keys()):
            for test_variable in sorted(result_table.keys()):
                result = [float(r[key]) for r in result_table[test_variable]]
                score = OrderedDict({'mean': float('{0:.2f}'.format(mean(result))),
                                     'std': float('{0:.2f}'.format(std(result))),
                                     'max': float('{0:.2f}'.format(max(result))),
                                     'min': float('{0:.2f}'.format(min(result)))})
                print(key.ljust(25), test_variable, re.compile('\((.*)\)').findall(str(score))[0])
            print('')

    @staticmethod
    def _save_as_csv(result_table, test_name, file_name):
        base_dir = os.path.dirname(__file__)
        dir_path = os.path.join(base_dir, '../log/clustering_log/AgglomerativeClustering/' + test_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(os.path.join(dir_path, file_name + '.csv'), 'w', encoding='utf8', newline='') as file:
            first_key = list(result_table.keys())[0]
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([''] + sorted(result_table[first_key][0].keys()))
            for compute in [mean, max, min, std]:
                writer.writerow(re.compile('function (\w*)').findall(str(compute)))
                for test_variable in sorted(result_table.keys()):
                    row_data = [test_variable]
                    for key in sorted(result_table[first_key][0].keys()):
                        result = [float(r[key]) for r in result_table[test_variable]]
                        row_data += [float('{0:.2f}'.format(compute(result)))]
                    writer.writerow(row_data)

            writer.writerow(['test finished in {0:.2f} seconds'.format(time.time() - start_time)])


def test_tf():
    t = AgglomerativeClusteringTester(FEATURE_TF, number_article_per_test_cluster=10)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=False, start_th=0.3, end_th=0.65, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.3, end_th=0.6, times=2)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)

    args = [(HAC.LINKAGE_CENTROID, 0.55), (HAC.LINKAGE_COMPLETE, 0.45),
            (HAC.LINKAGE_AVERAGE, 0.3), (HAC.LINKAGE_SINGLE, 0.2)]
    t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=3)


def test_tf_idf():
    t = AgglomerativeClusteringTester(FEATURE_TF_IDF, number_article_per_test_cluster=50)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)

    # content
    # args = [(HAC.LINKAGE_CENTROID, 0.25), (HAC.LINKAGE_COMPLETE, 0.2),
    #         (HAC.LINKAGE_AVERAGE, 0.1), (HAC.LINKAGE_SINGLE, 0.05)]
    # title
    args = [(HAC.LINKAGE_CENTROID, 0.05), (HAC.LINKAGE_COMPLETE, 0.3),
            (HAC.LINKAGE_AVERAGE, 0.05), (HAC.LINKAGE_SINGLE, 0.05)]
    t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=5)


def test_article():
    # t = AgglomerativeClusteringTester(FEATURE_ARTICLE, number_article_per_test_cluster=50)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.65, end_th=0.9, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=0.95, times=5)

    args = [(HAC.LINKAGE_CENTROID, 0.9), (HAC.LINKAGE_COMPLETE, 0.9),
            (HAC.LINKAGE_AVERAGE, 0.85), (HAC.LINKAGE_SINGLE, 0.75)]
    # t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=5)

    # use idf
    t = AgglomerativeClusteringTester(FEATURE_ARTICLE, number_article_per_test_cluster=10, use_idf=True)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.65, end_th=0.9, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=0.95, times=5)

    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_DOT, quick=True, start_th=0.75, end_th=1, times=5)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_DOT, quick=True, start_th=0.65, end_th=0.9, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_DOT, quick=True, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_DOT, quick=True, start_th=0.75, end_th=0.95, times=5)
    args = [(HAC.LINKAGE_CENTROID, 0.85), (HAC.LINKAGE_COMPLETE, 0.9),
            (HAC.LINKAGE_AVERAGE, 0.85), (HAC.LINKAGE_SINGLE, 0.75)]
    t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=5)


def test_extraction():
    t = AgglomerativeClusteringTester(FEATURE_ARTICLE_EXTRACTION, number_article_per_test_cluster=50)
    # t = AgglomerativeClusteringTester(FEATURE_ARTICLE_EXTRACTION, number_article_per_test_cluster=10, use_idf=True)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.4, end_th=0.6, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.7, end_th=0.95, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.5, end_th=0.8, times=5)
    args = [(HAC.LINKAGE_CENTROID, 0.8), (HAC.LINKAGE_COMPLETE, 0.8),
            (HAC.LINKAGE_AVERAGE, 0.65), (HAC.LINKAGE_SINGLE, 0.5)]
    # t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=5)

    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=True, start_th=0.7, end_th=0.95, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=True, start_th=0.4, end_th=0.6, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=True, start_th=0.7, end_th=0.95, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=True, start_th=0.5, end_th=0.75, times=5)
    args = [(HAC.LINKAGE_CENTROID, 0.8), (HAC.LINKAGE_COMPLETE, 0.8), (HAC.LINKAGE_AVERAGE, 0.6),
            (HAC.LINKAGE_AVERAGE, 0.65), (HAC.LINKAGE_SINGLE, 0.45)]
    # t.compare(HAC.SIMILARITY_COSINE, quick=True, args=args, sampling=True, times=5)

    # k = 5
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_DOT, quick=True, start_th=0.3, end_th=0.7, times=4)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_DOT, quick=True, start_th=0.1, end_th=0.5, times=4)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_DOT, quick=True, start_th=0.5, end_th=0.8, times=4)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_DOT, quick=True, start_th=0.3, end_th=0.7, times=4)
    # k = 10
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_DOT, quick=True, start_th=0.3, end_th=0.8, times=6)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_DOT, quick=True, start_th=0.4, end_th=0.6, times=6)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_DOT, quick=True, start_th=0.7, end_th=0.95, times=6)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_DOT, quick=True, start_th=0.5, end_th=0.8, times=6)
    # k =15
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_DOT, quick=True, start_th=0.55, end_th=0.95, times=7)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_DOT, quick=True, start_th=0.3, end_th=0.7, times=7)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_DOT, quick=True, start_th=0.6, end_th=0.9, times=7)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_DOT, quick=True, start_th=0.55, end_th=0.85, times=7)

    # lda k=5
    # args = [(HAC.LINKAGE_CENTROID, 0.55), (HAC.LINKAGE_COMPLETE, 0.7),
    #         (HAC.LINKAGE_AVERAGE, 0.55), (HAC.LINKAGE_SINGLE, 0.35)]
    # lda k=5 idf
    # args = [(HAC.LINKAGE_CENTROID, 0.55), (HAC.LINKAGE_COMPLETE, 0.7),
    #         (HAC.LINKAGE_AVERAGE, 0.55), (HAC.LINKAGE_SINGLE, 0.45)]
    # t.compare(HAC.SIMILARITY_DOT, quick=True, args=args, sampling=True, times=5)
    # tfidf k=5
    # args = [(HAC.LINKAGE_CENTROID, 0.55), (HAC.LINKAGE_CENTROID, 0.6), (HAC.LINKAGE_COMPLETE, 0.75),
    #         (HAC.LINKAGE_AVERAGE, 0.55), (HAC.LINKAGE_AVERAGE, 0.6), (HAC.LINKAGE_SINGLE, 0.35)]
    # tfidf k=5 idf
    # args = [(HAC.LINKAGE_CENTROID, 0.5), (HAC.LINKAGE_CENTROID, 0.55), (HAC.LINKAGE_COMPLETE, 0.65),
    #         (HAC.LINKAGE_AVERAGE, 0.5), (HAC.LINKAGE_AVERAGE, 0.55), (HAC.LINKAGE_SINGLE, 0.5)]
    # t.compare(HAC.SIMILARITY_DOT, quick=True, args=args, sampling=True, times=6)
    # k=10
    # args = [(HAC.LINKAGE_CENTROID, 0.60), (HAC.LINKAGE_CENTROID, 0.65), (HAC.LINKAGE_COMPLETE, 0.8),
    #         (HAC.LINKAGE_AVERAGE, 0.60), (HAC.LINKAGE_AVERAGE, 0.65), (HAC.LINKAGE_SINGLE, 0.5)]
    # lda k=15
    # args = [(HAC.LINKAGE_CENTROID, 0.65), (HAC.LINKAGE_CENTROID, 0.7), (HAC.LINKAGE_COMPLETE, 0.8),
    #         (HAC.LINKAGE_AVERAGE, 0.7), (HAC.LINKAGE_SINGLE, 0.6)]
    # lda k=15 weight
    # args = [(HAC.LINKAGE_CENTROID, 0.65), (HAC.LINKAGE_CENTROID, 0.7), (HAC.LINKAGE_COMPLETE, 0.8),
    #         (HAC.LINKAGE_AVERAGE, 0.7), (HAC.LINKAGE_SINGLE, 0.5)]
    # lda tfidf k=20 weight
    # args = [(HAC.LINKAGE_CENTROID, 0.7), (HAC.LINKAGE_COMPLETE, 0.85),
    #         (HAC.LINKAGE_AVERAGE, 0.7), (HAC.LINKAGE_SINGLE, 0.55)]
    # lda k=25 weight
    # args = [(HAC.LINKAGE_CENTROID, 0.75), (HAC.LINKAGE_COMPLETE, 0.85),
    #         (HAC.LINKAGE_AVERAGE, 0.75), (HAC.LINKAGE_SINGLE, 0.55)]
    # lda k=30 weight
    # args = [(HAC.LINKAGE_CENTROID, 0.75), (HAC.LINKAGE_COMPLETE, 0.85),
    #         (HAC.LINKAGE_AVERAGE, 0.75), (HAC.LINKAGE_SINGLE, 0.6)]
    # t.compare(HAC.SIMILARITY_DOT, quick=True, args=args, sampling=True, times=5)
    # tfidf k=15
    # args = [(HAC.LINKAGE_CENTROID, 0.7), (HAC.LINKAGE_COMPLETE, 0.8),
    #         (HAC.LINKAGE_AVERAGE, 0.7), (HAC.LINKAGE_SINGLE, 0.55)]
    # tfidf weight k = 15
    args = [(HAC.LINKAGE_CENTROID, 0.65), (HAC.LINKAGE_COMPLETE, 0.8),
            (HAC.LINKAGE_AVERAGE, 0.65), (HAC.LINKAGE_SINGLE, 0.55)]
    t.compare(HAC.SIMILARITY_DOT, quick=True, args=args, sampling=True, times=6)


def test_title():
    t = AgglomerativeClusteringTester(FEATURE_TITLE, number_article_per_test_cluster=50)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_DOT, quick=False, start_th=0.4, end_th=0.75, times=5)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_DOT, quick=False, start_th=0.1, end_th=0.6, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_DOT, quick=False, start_th=0.5, end_th=0.95, times=5)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_DOT, quick=False, start_th=0.35, end_th=0.85, times=5)

    # COS HAC
    # args = [(HAC.LINKAGE_CENTROID, 0.65), (HAC.LINKAGE_COMPLETE, 0.75),
    #         (HAC.LINKAGE_AVERAGE, 0.55), (HAC.LINKAGE_SINGLE, 0.45)]
    # t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=5)
    # DOT QUCIK
    # args = [(HAC.LINKAGE_CENTROID, 0.55), (HAC.LINKAGE_CENTROID, 0.6), (HAC.LINKAGE_COMPLETE, 0.75),
    #         (HAC.LINKAGE_AVERAGE, 0.6), (HAC.LINKAGE_SINGLE, 0.5)]
    # t.compare(HAC.SIMILARITY_DOT, quick=True, args=args, sampling=True, times=5)
    # DOT HAC
    # args = [(HAC.LINKAGE_CENTROID, 0.55), (HAC.LINKAGE_COMPLETE, 0.75),
    #         (HAC.LINKAGE_AVERAGE, 0.55), (HAC.LINKAGE_SINGLE, 0.45)]
    # t.compare(HAC.SIMILARITY_DOT, quick=False, args=args, sampling=True, times=5)
    # COS QUICK
    # args = [(HAC.LINKAGE_CENTROID, 0.75), (HAC.LINKAGE_COMPLETE, 0.75),
    #         (HAC.LINKAGE_AVERAGE, 0.6), (HAC.LINKAGE_SINGLE, 0.45)]
    # t.compare(HAC.SIMILARITY_COSINE, quick=True, args=args, sampling=True, times=5)


def test_title_extraction():
    t = AgglomerativeClusteringTester(FEATURE_TITLE_EXTRACTION, number_article_per_test_cluster=50)
    # t.find_ratio_threshold(1, 15, 0.1, 0.9, times=5)
    # t.find_ratio_threshold(1, 15, 0.2, 0.8, times=5)
    # t.find_ratio_threshold(1, 15, 0.3, 0.7, times=5)
    # t.find_ratio_threshold(1, 15, 0.4, 0.6, times=5)
    # t.find_ratio_threshold(1, 15, 0.5, 0.5, times=5)
    # t.find_ratio_threshold(1, 15, 0.6, 0.4, times=5)
    # t.find_ratio_threshold(1, 15, 0.7, 0.3, times=5)
    # t.find_ratio_threshold(1, 15, 0.8, 0.2, times=5)
    # t.find_ratio_threshold(1, 15, 0.9, 0.1, times=5)


def compare_all():
    # feature, linkage, threshold, similarity_tpye, quick, weight

    # args = [
        # (FEATURE_TF, HAC.LINKAGE_SINGLE, 0.2, HAC.SIMILARITY_COSINE, False, False),
        # (FEATURE_TF_IDF, HAC.LINKAGE_AVERAGE, 0.1, HAC.SIMILARITY_COSINE, False, False),
        # (FEATURE_ARTICLE, HAC.LINKAGE_SINGLE, 0.75, HAC.SIMILARITY_COSINE, False, False),
        # (FEATURE_ARTICLE, HAC.LINKAGE_SINGLE, 0.75, HAC.SIMILARITY_COSINE, False, True),
        # (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.65, HAC.SIMILARITY_COSINE, False, False),
        # (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.65, HAC.SIMILARITY_COSINE, True, False),
        # (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.65, HAC.SIMILARITY_DOT, True, False),
        # (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_CENTROID, 0.65, HAC.SIMILARITY_DOT, True, False),
        # (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_CENTROID, 0.65, HAC.SIMILARITY_DOT, False, False),
        # (FEATURE_TITLE, HAC.LINKAGE_CENTROID, 0.55, HAC.SIMILARITY_DOT, True)
    # ]
    args = [
        (FEATURE_TF_IDF, HAC.LINKAGE_AVERAGE, 0.05, HAC.SIMILARITY_COSINE, False, False),
        (FEATURE_TITLE, HAC.LINKAGE_AVERAGE, 0.55, HAC.SIMILARITY_COSINE, False, False)
    ]
    t = AgglomerativeClusteringTester(FEATURE_TITLE, number_article_per_test_cluster=50)
    t.compare_different_method('compare_all', args, sampling=True, times=3)


def compare_speed():
    args = [
        (FEATURE_TITLE, HAC.LINKAGE_AVERAGE, 0.55, HAC.SIMILARITY_COSINE, False, False),
        (FEATURE_TITLE, HAC.LINKAGE_AVERAGE, 0.6, HAC.SIMILARITY_COSINE, True, False),
        (FEATURE_TITLE, HAC.LINKAGE_CENTROID, 0.55, HAC.SIMILARITY_DOT, False, False),
        (FEATURE_TITLE, HAC.LINKAGE_CENTROID, 0.55, HAC.SIMILARITY_DOT, True, False)
    ]
    t = AgglomerativeClusteringTester(FEATURE_TITLE, number_article_per_test_cluster=50)
    t.compare_different_method('compare_speed', args, sampling=True, times=3)


def compare_time_feature():
    t = AgglomerativeClusteringTester(FEATURE_TITLE, number_article_per_test_cluster=50)
    t.compare_time_feature('compare time feature', 0.55, HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, sampling=True,
                           times=3)


def compare_quick():
    args = [
        (FEATURE_TITLE, HAC.LINKAGE_AVERAGE, 0.55, HAC.SIMILARITY_COSINE, False, False),
        (FEATURE_TITLE, HAC.LINKAGE_AVERAGE, 0.6, HAC.SIMILARITY_COSINE, True, False),
    ]
    t = AgglomerativeClusteringTester(FEATURE_TITLE, number_article_per_test_cluster=50)
    t.compare_different_method('compare_quick', args, sampling=True, times=3)


def compare_extraction():
    args = [
        (0, 5, HAC.LINKAGE_CENTROID, 0.55, False),
        (0, 10, HAC.LINKAGE_CENTROID, 0.6, False),
        (0, 10, HAC.LINKAGE_CENTROID, 0.65, False),
        (0, 15, HAC.LINKAGE_CENTROID, 0.7, False),
        (1, 5, HAC.LINKAGE_CENTROID, 0.55, False),
        (1, 10, HAC.LINKAGE_CENTROID, 0.6, False),
        (1, 15, HAC.LINKAGE_CENTROID, 0.7, False),
    ]
    weight_args = [
        (0, 10, HAC.LINKAGE_CENTROID, 0.65, True),
        (0, 15, HAC.LINKAGE_CENTROID, 0.65, True),
        (0, 20, HAC.LINKAGE_CENTROID, 0.7, True),
        (0, 25, HAC.LINKAGE_CENTROID, 0.75, True),
        (0, 30, HAC.LINKAGE_CENTROID, 0.75, True),
        (1, 10, HAC.LINKAGE_CENTROID, 0.6, True),
        (1, 15, HAC.LINKAGE_CENTROID, 0.65, True),
        (1, 20, HAC.LINKAGE_CENTROID, 0.7, True),
        (1, 25, HAC.LINKAGE_CENTROID, 0.7, True),
        (1, 25, HAC.LINKAGE_CENTROID, 0.75, True),
        (1, 30, HAC.LINKAGE_CENTROID, 0.75, True),
    ]
    t = AgglomerativeClusteringTester(FEATURE_ARTICLE_EXTRACTION, number_article_per_test_cluster=50)
    # t.compare_extraction(args, sampling=True, times=3)
    t.compare_extraction(weight_args, sampling=True, times=3)


def compare_ratio():
    t = AgglomerativeClusteringTester(FEATURE_TITLE_EXTRACTION, number_article_per_test_cluster=50)
    # t, c, threshold
    args = [
        (0.9, 0.1, 0.6),
        (0.8, 0.2, 0.6),
        (0.7, 0.3, 0.6),
        (0.6, 0.4, 0.65),
        (0.5, 0.5, 0.65),
        (0.4, 0.6, 0.65)
    ]
    t.compare_ratio(1, 15, args, sampling=True, times=3)

if __name__ == '__main__':
    start_time = time.time()
    # test_title()
    # test_tf()
    # test_tf_idf()
    # test_article()
    # test_extraction()
    # test_title_extraction()
    # compare_all()
    # compare_speed()
    # compare_time_feature()
    # compare_extraction()
    compare_ratio()
    print('test finished in {0:.2f} seconds'.format(time.time() - start_time))
