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

FEATURE_TITLE = 'title'
FEATURE_ARTICLE = 'whole article'
FEATURE_ARTICLE_KEY_WORD_EXTRACTION = 'article with keyword extraction'


class AgglomerativeClusteringTester:
    def __init__(self, feature, model_path='model/bin/ngram_300_3_83w.bin'):
        self._feature_extractor = FeatureExtractor(model_path=model_path)
        self._all_test_clusters = test_data.get_test_clusters()
        self._feature_mode = feature

        for cluster in self._all_test_clusters:
            if self._feature_mode == FEATURE_TITLE:
                self._feature_extractor.fit_with_extraction_ratio(articles=cluster['articles'], t=1, c=0)
            elif self._feature_mode == FEATURE_ARTICLE:
                self._feature_extractor.fit(articles=cluster['articles'])
            elif self._feature_mode == FEATURE_ARTICLE_KEY_WORD_EXTRACTION:
                self._feature_extractor.fit_with_extraction(articles=cluster['articles'])
            else:
                raise ValueError('Feature not assign yet')
        print('tester init with feature', self._feature_mode)

    def _get_test_articles(self, sampling=True):
        articles = []
        self._labeled_clusters = []
        for cluster in self._all_test_clusters:
            cluster_articles = cluster['articles']
            if sampling:
                cluster_articles = random.sample(cluster_articles, k=random.randint(1, len(cluster_articles)))
            self._labeled_clusters.append({'articles': cluster_articles})
            articles.extend(cluster_articles)
        random.shuffle(articles)
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

    def find_best_threshold(self, linkage, sim, quick, start_th=0.3, end_th=0.8, step=0.1, sampling=True, times=1):
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

    def compare(self, sampling=False, times=1):
        file_name = 'compare sampling={} times={}'.format(sampling, times)
        print(file_name)
        result_table = {}
        args = {HAC.LINKAGE_CENTROID: 0.55, HAC.LINKAGE_COMPLETE: 0.55, HAC.LINKAGE_AVERAGE: 0.55,
                HAC.LINKAGE_SINGLE: 0.55}
        for time_counter in range(times):
            print(time_counter)
            articles = self._get_test_articles(sampling)
            for linkage, threshold in args.items():
                clusters = HAC(threshold, linkage=linkage).quick_fit(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                key = '{}, {0:.2f}'.format(linkage, threshold)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)

        self._print_test_result(result_table)
        self._save_as_csv(result_table, self._feature_mode, file_name)

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


if __name__ == '__main__':
    start_time = time.time()
    tester = AgglomerativeClusteringTester(FEATURE_ARTICLE)

    # tester.stable_test()
    # tester.find_best_threshold(HAC.LINKAGE_CENTROID, HAC.SIMILARITY_COSINE, quick=True, times=25, step=0.05)
    # tester.find_best_threshold(HAC.LINKAGE_SINGLE, HAC.SIMILARITY_COSINE, quick=True, times=5, step=0.05)
    # tester.find_best_threshold(HAC.LINKAGE_COMPLETE, HAC.SIMILARITY_COSINE, quick=True, times=25, step=0.05)
    # tester.find_best_threshold(HAC.LINKAGE_AVERAGE, HAC.SIMILARITY_COSINE, quick=True, times=25, step=0.05)
    print('test finished in {0:.2f} seconds'.format(time.time() - start_time))
