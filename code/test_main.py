import random
import re
from collections import OrderedDict

import code.test.make_test_data as test_data
import code.main as main

from numpy import mean, std
import time


class MainTester:
    def __init__(self, model_path='model/bin/ngram_300_3_83w.bin'):
        self._model = main.load_model(model_path=model_path)
        self._all_test_clusters = test_data.get_test_clusters()

        for cluster in self._all_test_clusters:
            main.compute_article_vector(self._model, cluster['articles'])

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

    # algorithm 2 cost 6 seconds per times on 183.28
    def find_best_threshold(self, algorithm, start_th=0.4, end_th=0.65, increase_count=0.5, sampling=False, times=1):
        result_table = {}
        for time_counter in range(times):
            articles = self._get_test_articles(sampling)
            threshold = start_th
            while threshold < end_th:
                clusters = algorithm(self._model, articles, threshold)
                result = main.validate_clustering(self._labeled_clusters, clusters)
                key = '{0:.2f}'.format(threshold)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
                threshold += increase_count
        self._print_test_result(result_table)

    # algorithm 3 cost 482 seconds per times on 183.28
    def find_best_ratio_between_title_and_content(self, algorithm, sampling=True, times=1):
        result_table = {}
        for time_counter in range(times):
            articles = self._get_test_articles(sampling)
            for t in range(5, 10):
                t_ratio = t / 10
                c_ratio = (10 - t) / 10
                clusters = algorithm(self._model, articles, 0.6, t_ratio, c_ratio)
                result = main.validate_clustering(self._labeled_clusters, clusters)
                if t_ratio not in result_table:
                    result_table[t_ratio] = []
                result_table[t_ratio].append(result)
        self._print_test_result(result_table)

    # cost 474 seconds per times on 183.28
    def compare_clustering(self, sampling=True, times=1):
        result_table = {}
        threshold = 0.6
        for time_counter in range(times):
            articles = self._get_test_articles(sampling)
            for algorithm in [main.clustering1, main.clustering2, main.clustering3]:
                if algorithm is main.clustering2:
                    clusters = main.clustering2(self._model, articles, threshold)
                else:
                    clusters = algorithm(self._model, articles, threshold, t=0.8, c=0.2)
                result = main.validate_clustering(self._labeled_clusters, clusters)
                algorithm_name = str(algorithm).split(' ')[1]
                if algorithm_name not in result_table:
                    result_table[algorithm_name] = []
                result_table[algorithm_name].append(result)
        self._print_test_result(result_table)

    @staticmethod
    def _print_test_result(result_table):
        first_key = list(result_table.keys())[0]
        for key in sorted(result_table[first_key][0].keys()):
            for algorithm in sorted(result_table.keys()):
                result = [float(r[key]) for r in result_table[algorithm]]
                score = OrderedDict({'mean': float('{0:.2f}'.format(mean(result))),
                                     'std': float('{0:.2f}'.format(std(result))),
                                     'max': float('{0:.2f}'.format(max(result))),
                                     'min': float('{0:.2f}'.format(min(result)))})
                print(key.ljust(25), algorithm, re.compile('\((.*)\)').findall(str(score))[0])
            print('')


if __name__ == '__main__':
    start_time = time.time()
    tester = MainTester()

    # tester.compare_clustering()
    # tester.find_best_ratio_between_title_and_content(main.clustering3, sampling=False)
    tester.find_best_threshold(main.clustering2, 0.45, 0.6, 0.01, False, 1)
    # find_best_threshold(model, 2, False, 0.45, 21, 0.01, 5)

    print('test finished in {0:.2f} seconds'.format(time.time() - start_time))
