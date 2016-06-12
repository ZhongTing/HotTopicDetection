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


class AgglomerativeClusteringTester:
    def __init__(self, feature, model_path='model/bin/ngram_300_3_83w.bin', number_article_per_test_cluster=None):
        if feature != FEATURE_TF_IDF and feature != FEATURE_TF:
            self._feature_extractor = FeatureExtractor(model_path=model_path)
        self._all_test_clusters = test_data.get_test_clusters()
        self._feature_mode = feature
        self._number_article_per_test_cluster = number_article_per_test_cluster

        if self._feature_mode == FEATURE_TF_IDF:
            pass
        else:
            for cluster in self._all_test_clusters:
                self._feature_extraction(self._feature_mode, cluster['articles'])
        print('tester init with feature', self._feature_mode)

    def _feature_extraction(self, feature, articles):
        if feature == FEATURE_TF:
            TFIDFFeatureExtractor(use_idf=False).fit(articles)
        elif feature == FEATURE_TF_IDF:
            TFIDFFeatureExtractor(use_idf=True).fit(articles)
        elif feature == FEATURE_ARTICLE:
            self._feature_extractor.fit(articles)
        elif feature == FEATURE_ARTICLE_EXTRACTION:
            self._feature_extractor.fit_with_extraction(articles)
        elif self._feature_mode == FEATURE_TITLE:
            self._feature_extractor.fit_with_extraction_ratio(articles, t=1, c=0)
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
        if self._feature_mode == FEATURE_TF or self._feature_mode == FEATURE_TF_IDF:
            self._feature_extraction(self._feature_mode, articles)
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
        file_name = 'threshold {} {} quick={} sampling={} times={}'.format(linkage, sim, quick, sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            threshold = start_th
            while threshold < end_th + step:
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

    def compare_different_method(self, args, sampling=False, times=1):
        file_name = 'compare all {} sampling={} times={}'.format(self._number_article_per_test_cluster, sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            for (feature, linkage, threshold, sim, quick) in args:
                t = time.time()
                self._feature_extraction(feature, articles)
                if quick:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).quick_fit(articles)
                else:
                    clusters = HAC(threshold, linkage=linkage, similarity=sim).fit(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                result['time'] = time.time() - t
                key = '{} {} {} {} {}'.format(feature, linkage, threshold, sim, quick)
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
    t = AgglomerativeClusteringTester(FEATURE_TF_IDF, number_article_per_test_cluster=10)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.05, end_th=0.4, times=3)

    args = [(HAC.LINKAGE_CENTROID, 0.25), (HAC.LINKAGE_COMPLETE, 0.2),
            (HAC.LINKAGE_AVERAGE, 0.1), (HAC.LINKAGE_SINGLE, 0.05)]
    t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=5)


def test_article():
    t = AgglomerativeClusteringTester(FEATURE_ARTICLE, number_article_per_test_cluster=50)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.65, end_th=0.9, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=0.95, times=5)

    args = [(HAC.LINKAGE_CENTROID, 0.9), (HAC.LINKAGE_COMPLETE, 0.9),
            (HAC.LINKAGE_AVERAGE, 0.85), (HAC.LINKAGE_SINGLE, 0.75)]
    t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=5)


def test_extraction():
    t = AgglomerativeClusteringTester(FEATURE_ARTICLE_EXTRACTION, number_article_per_test_cluster=50)
    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=False, start_th=0.75, end_th=1, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.4, end_th=0.6, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.7, end_th=0.95, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=False, start_th=0.5, end_th=0.8, times=5)
    args = [(HAC.LINKAGE_CENTROID, 0.8), (HAC.LINKAGE_COMPLETE, 0.8),
            (HAC.LINKAGE_AVERAGE, 0.65), (HAC.LINKAGE_SINGLE, 0.5)]
    t.compare(HAC.SIMILARITY_COSINE, quick=False, args=args, sampling=True, times=5)

    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=True, start_th=0.7, end_th=0.95, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=True, start_th=0.4, end_th=0.6, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=True, start_th=0.7, end_th=0.95, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=True, start_th=0.5, end_th=0.75, times=5)
    args = [(HAC.LINKAGE_CENTROID, 0.8), (HAC.LINKAGE_COMPLETE, 0.8), (HAC.LINKAGE_AVERAGE, 0.6),
            (HAC.LINKAGE_AVERAGE, 0.65), (HAC.LINKAGE_SINGLE, 0.45)]
    # t.compare(HAC.SIMILARITY_COSINE, quick=True, args=args, sampling=True, times=5)

    # t.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_DOT, quick=True, start_th=0.5, end_th=0.8, times=3)
    # t.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_DOT, quick=True, start_th=0.4, end_th=0.6, times=5)
    # t.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_DOT, quick=True, start_th=0.7, end_th=0.95, times=3)
    # t.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_DOT, quick=True, start_th=0.5, end_th=0.8, times=5)
    args = [(HAC.LINKAGE_CENTROID, 0.65), (HAC.LINKAGE_COMPLETE, 0.8),
            (HAC.LINKAGE_AVERAGE, 0.65), (HAC.LINKAGE_SINGLE, 0.5)]
    # t.compare(HAC.SIMILARITY_DOT, quick=True, args=args, sampling=True, times=5)


def compare_all():
    args = [(FEATURE_TF, HAC.LINKAGE_SINGLE, 0.2, HAC.SIMILARITY_COSINE, False),
            (FEATURE_TF_IDF, HAC.LINKAGE_AVERAGE, 0.1, HAC.SIMILARITY_COSINE, False),
            (FEATURE_ARTICLE, HAC.LINKAGE_SINGLE, 0.75, HAC.SIMILARITY_COSINE, False),
            (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.45, HAC.SIMILARITY_COSINE, False),
            (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.65, HAC.SIMILARITY_COSINE, True),
            (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.65, HAC.SIMILARITY_DOT, True)]
    t = AgglomerativeClusteringTester(FEATURE_TITLE, number_article_per_test_cluster=50)
    t.compare_different_method(args, sampling=True, times=3)


def compare_speed():
    args = [(FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.45, HAC.SIMILARITY_COSINE, False),
            (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_CENTROID, 0.8, HAC.SIMILARITY_COSINE, False),
            (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.65, HAC.SIMILARITY_COSINE, True),
            (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_CENTROID, 0.8, HAC.SIMILARITY_COSINE, True),
            (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_AVERAGE, 0.65, HAC.SIMILARITY_DOT, True),
            (FEATURE_ARTICLE_EXTRACTION, HAC.LINKAGE_CENTROID, 0.65, HAC.SIMILARITY_DOT, True)]
    t = AgglomerativeClusteringTester(FEATURE_TITLE, number_article_per_test_cluster=50)
    t.compare_different_method(args, sampling=True, times=3)


if __name__ == '__main__':
    start_time = time.time()
    # test_tf()
    # test_tf_idf()
    # test_article()
    # test_extraction()
    # compare_all()
    # compare_speed()
    print('test finished in {0:.2f} seconds'.format(time.time() - start_time))
