import random
import re
from collections import OrderedDict

import python_code.test.make_test_data as test_data
# noinspection PyPep8Naming
from python_code.clustering_v2.agglomerative_clustering import AgglomerativeClustering as HAC
import python_code.feature_extractor as extractor

from numpy import mean, std
import time
import os
import csv

from python_code.clustering_validation import validate_clustering

FEATURE_TF = 'tf whole article'
FEATURE_TF_IDF = 'tfidf whole article'
FEATURE_ARTICLE = 'whole article'
FEATURE_ARTICLE_EXTRACTION = 'article with keyword extraction'
FEATURE_TITLE = 'title'
FEATURE_TITLE_EXTRACTION = 'title with keyword extraction'


class AgglomerativeClusteringTester:
    def __init__(self):
        # self._file_name = '20160615.json'
        self._file_name = '20160624.json'
        print(self._file_name)
        self._labeled_clusters = test_data.get_test_clusters(self._file_name)

    def _get_test_articles(self):
        articles = [article for cluster in self._labeled_clusters for article in cluster['articles']]
        random.shuffle(articles)
        return articles

    def get_article_with_feature_extraction(self, feature_extractor):
        if isinstance(feature_extractor, extractor.TFIDF):
            articles = self._get_test_articles()
            feature_extractor.fit(articles)
        else:
            for cluster in self._labeled_clusters:
                feature_extractor.fit(cluster['articles'])
            articles = self._get_test_articles()
        return articles

    def best_threshold(self, feature_extractor, linkage, similarity, start_th, end_th, step):
        file_name = 'best_threshold {} {} {}'.format(feature_extractor.args(), linkage, similarity)
        articles = self.get_article_with_feature_extraction(feature_extractor)
        result_table = {}
        threshold = start_th
        try:
            while threshold < end_th + step:
                print(threshold)
                t = time.time()
                clusters = HAC(threshold=threshold, linkage=linkage, similarity=similarity).fit(articles)
                t = time.time() - t
                result = validate_clustering(self._labeled_clusters, clusters)
                result['time'] = t
                key = '{0:.2f}'.format(threshold)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
                # if float(result['AMI']) < 0.1:
                #     threshold += step
                threshold += step

            self._print_test_result(result_table)
            self._save_as_csv(result_table, os.path.join(feature_extractor.name(), self._file_name), file_name)
        except ValueError as e:
            print(e)
            return
        else:
            return

    def stable_test(self, feature_extractor, threshold, linkage, similarity, times):
        file_name = 'stable_test {} {} {} {}'.format(feature_extractor.args(), threshold, linkage, similarity)
        articles = self.get_article_with_feature_extraction(feature_extractor)
        result_table = {}
        for i in range(times):
            print('stable_test', i)
            clusters = HAC(threshold=threshold, linkage=linkage, similarity=similarity).fit(articles)
            result = validate_clustering(self._labeled_clusters, clusters)
            key = '{0:.2f}'.format(threshold)
            if key not in result_table:
                result_table[key] = []
            result_table[key].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, 'stable_test', file_name)

    def time_test(self, args):
        result_table = {}
        for k in range(100, 1001, 100):
            print('k', k)
            result = {}
            for (e, linkage, similarity, threshold) in args:
                articles = self._get_test_articles()
                if k < len(articles):
                    articles = random.sample(articles, k)
                t = time.time()
                e.fit(articles)
                HAC(threshold=threshold, linkage=linkage, similarity=similarity).fit(articles)
                result[e.name() + e.args()] = time.time() - t
                key = str(k)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, 'time_test', self._file_name)

    def print_data_set(self):
        result_table = {}
        file_name = self._file_name
        for cluster in sorted(self._labeled_clusters, key=lambda a: len(a['articles']), reverse=True):
            key = len(cluster['articles'])
            if key not in result_table:
                result_table[key] = [{'count': 0}]
            result_table[key][0]['count'] += 1
        self._save_as_csv(result_table, 'data_set', file_name)

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


def idf():
    print('idf')
    tester = AgglomerativeClusteringTester()
    for only_title in [True, False]:
        feature_extractor = extractor.TFIDF(use_idf=False, only_title=only_title)
        for linkage in [HAC.LINKAGE_CENTROID, HAC.LINKAGE_COMPLETE, HAC.LINKAGE_SINGLE, HAC.LINKAGE_AVERAGE]:
            print(linkage, 'only title', only_title)
            tester.best_threshold(feature_extractor, linkage, HAC.SIMILARITY_COSINE, 0.05, 0.4, step=0.05)


def title():
    print('title')
    tester = AgglomerativeClusteringTester()
    model = extractor.load_model('model/bin/ngram_300_5_90w.bin')
    feature_extractor = extractor.Title(model)
    for linkage in [HAC.LINKAGE_CENTROID, HAC.LINKAGE_COMPLETE, HAC.LINKAGE_SINGLE, HAC.LINKAGE_AVERAGE]:
        print(linkage)
        tester.best_threshold(feature_extractor, linkage, HAC.SIMILARITY_DOT, 0.6, 0.9, step=0.05)


def extraction():
    print('extraction')
    tester = AgglomerativeClusteringTester()
    model = extractor.load_model('model/bin/ngram_300_5_90w.bin')
    for k in [30]:
        for with_weight in [True]:
            for method in [1]:
                feature_extractor = extractor.ContentExtraction(model, method, k, with_weight=with_weight)
                for linkage in [HAC.LINKAGE_CENTROID]:
                    for sim in [HAC.SIMILARITY_DOT]:
                        print(with_weight, k, method, linkage, sim)
                        tester.best_threshold(feature_extractor, linkage, sim, 0.5, 0.95, step=0.05)


def stable_test():
    print('extraction')
    tester = AgglomerativeClusteringTester()
    model = extractor.load_model('model/bin/ngram_300_5_90w.bin')
    feature_extractor = extractor.ContentExtraction(model, 0, 5, with_weight=True)
    tester.stable_test(feature_extractor, 0.75, HAC.LINKAGE_CENTROID, HAC.SIMILARITY_DOT, 5)


def ratio():
    print('ratio')
    tester = AgglomerativeClusteringTester()
    model = extractor.load_model('model/bin/ngram_300_5_90w.bin')
    r = [(0, 1), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
         (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1), (1, 0)]
    for i in range(len(r)):
        print(r[i])
        feature_extractor = extractor.ContentRatioExtraction(model, 1, 15, True, t_ratio=r[i][0], c_ratio=r[i][1])
        tester.best_threshold(feature_extractor, HAC.LINKAGE_CENTROID, HAC.SIMILARITY_DOT, 0.4, 0.9, step=0.05)


def time_test():
    print('time test')
    tester = AgglomerativeClusteringTester()
    model = extractor.load_model('model/bin/ngram_300_5_90w.bin')
    args = [
        (extractor.TFIDF(use_idf=True, only_title=False), HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, 0.15),
        (extractor.TFIDF(use_idf=True, only_title=True), HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, 0.1),
        (extractor.ContentRatioExtraction(model, 1, 15, True, t_ratio=0.5, c_ratio=0.5), HAC.LINKAGE_CENTROID,
         HAC.SIMILARITY_DOT, 0.65)
    ]
    tester.time_test(args)


if __name__ == '__main__':
    start_time = time.time()
    # idf()
    # title()
    # extraction()
    # ratio()
    # stable_test()
    # AgglomerativeClusteringTester().print_data_set()
    time_test()
    print('test finished in {0:.2f} seconds'.format(time.time() - start_time))
