import csv
import os
import random
import re
import time
from collections import OrderedDict

from numpy import mean, std

import python_code.clustering_v1.main as main
import python_code.model.ptt_article_fetcher as fetcher
import python_code.test.make_test_data as test_data


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
    def find_best_threshold(self, algorithm, start_th=0.4, end_th=0.65, step=0.05, sampling=False, times=1):
        algorithm_name = str(algorithm).split(' ')[1]
        file_name = 'threshold sampling={} times={}'.format(sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            articles = self._get_test_articles(sampling)
            threshold = start_th
            while threshold <= end_th:
                clusters = algorithm(self._model, articles, threshold)
                result = main.validate_clustering(self._labeled_clusters, clusters)
                key = '{0:.2f}'.format(threshold)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
                threshold += step
        self._print_test_result(result_table)
        self._save_as_csv(result_table, algorithm_name, file_name)

    # for algorithm 1, 4
    def find_best_threshold_with_ratio(self, algorithm, t=0.6, c=0.4, start_th=0.4, end_th=0.65, increase_count=0.5,
                                       sampling=False, times=1):
        algorithm_name = str(algorithm).split(' ')[1]
        file_name = 'threshold t={} c={} sampling={} times={}'.format(t * 10, c * 10, sampling, times)
        print(algorithm, file_name)
        result_table = {}
        for time_counter in range(times):
            articles = self._get_test_articles(sampling)
            threshold = start_th
            print('time counter', time_counter)
            while threshold <= end_th:
                clusters = algorithm(self._model, articles, threshold, t=t, c=c)
                result = main.validate_clustering(self._labeled_clusters, clusters)
                key = '{0:.2f}'.format(threshold)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
                threshold += increase_count
        self._print_test_result(result_table)
        self._save_as_csv(result_table, algorithm_name, file_name)

    def find_better_args_in_algorithm3(self, first_threshold, sampling=True, times=5):
        file_name = 'better_arg sampling={} times={}'.format(sampling, times)
        print('algorithm3', file_name)
        result_table = {}
        for time_counter in range(times):
            articles = self._get_test_articles(sampling)
            print('time counter', time_counter)
            for t in range(3, 10):
                t_ratio = float('{0:.2f}'.format(t / 10))
                c_ratio = float('{0:.2f}'.format(1 - t_ratio))
                for threshold in [0.45, 0.5, 0.55, 0.6, 0.65]:
                    clusters = main.clustering3(self._model, articles, first_threshold, threshold, t=t_ratio, c=c_ratio)
                    result = main.validate_clustering(self._labeled_clusters, clusters)
                    key = '{}:{}-{}'.format(int(t_ratio * 10), int(c_ratio * 10), threshold)
                    if key not in result_table:
                        result_table[key] = []
                    result_table[key].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, os.path.join('clustering3', str(first_threshold)), file_name)

    # algorithm 3 cost 482 seconds per times on 183.28
    def find_best_ratio(self, algorithm, args_set, sampling=True, times=1):
        algorithm_name = str(algorithm).split(' ')[1]
        file_name = 'ratio sampling={} times={}'.format(sampling, times)
        print(algorithm, file_name)
        result_table = {}
        for time_counter in range(times):
            articles = self._get_test_articles(sampling)
            print('time counter', time_counter)
            for (t_ratio, threshold) in args_set:
                c_ratio = float('{0:.2f}'.format(1 - t_ratio))
                key = '{}:{}-{}'.format(int(t_ratio * 10), int(c_ratio * 10), threshold)
                clusters = algorithm(self._model, articles, threshold, t=t_ratio, c=c_ratio)
                result = main.validate_clustering(self._labeled_clusters, clusters)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, algorithm_name, file_name)

    # cost 787 seconds per times on 183.3
    # cost 474 seconds per times on 183.28
    def compare_clustering(self, sampling=True, times=1):
        file_name = 'compare sampling={} times={}'.format(sampling, times)
        print(file_name)
        result_table = {}
        for time_counter in range(times):
            articles = self._get_test_articles(sampling)
            print('time counter', time_counter)
            for algorithm in [main.clustering1, main.clustering2, main.clustering3, main.clustering4]:
                clusters = algorithm(self._model, articles)
                result = main.validate_clustering(self._labeled_clusters, clusters, internal_validation=True)
                algorithm_name = str(algorithm).split(' ')[1]
                if algorithm_name not in result_table:
                    result_table[algorithm_name] = []
                result_table[algorithm_name].append(result)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, '', file_name)

    def compare_clustering_using_real_data(self, start_month='2016/06', days=30):
        file_name = 'compare_using_real_data {} days={}'.format(''.join(start_month.split('/')), days)
        print(file_name)
        result_table = {}
        days += 1
        for day_counter in range(1, days):
            target_day = '{}/{}'.format(start_month, str(day_counter).zfill(2))
            print(target_day)
            articles = fetcher.fetch_articles('*', 4000, end_day=target_day, days=0)
            if len(articles) is 0:
                continue
            main.compute_article_vector(self._model, articles)
            for algorithm in [main.clustering1, main.clustering2, main.clustering3, main.clustering4]:
                clusters = algorithm(self._model, articles)
                result = main.internal_validate(clusters)
                algorithm_name = str(algorithm).split(' ')[1]
                print(algorithm_name, result)
                if algorithm_name not in result_table:
                    result_table[algorithm_name] = []
                result_table[algorithm_name].append(result)

        self._print_test_result(result_table)
        self._save_as_csv(result_table, '', file_name)

    def test_time_complexity(self, start_size=20, stop_size=400, step=20):
        file_name = 'test_time_complexity {} {} {}'.format(start_size, stop_size, step)
        print(file_name)
        result_table = {}
        whole_articles = self._get_test_articles(sampling=False)
        for size in range(start_size, stop_size, step):
            articles = random.sample(whole_articles, k=size)
            print(size)
            result = {}
            for algorithm in [main.clustering1, main.clustering2, main.clustering3, main.clustering4]:
                algorithm_start_time = time.time()
                algorithm(self._model, articles)
                spend_time = time.time() - algorithm_start_time
                algorithm_name = str(algorithm).split(' ')[1]
                result[algorithm_name] = spend_time

            if size not in result_table:
                result_table[size] = []
            result_table[size].append(result)
        print(result_table)
        self._print_test_result(result_table)
        self._save_as_csv(result_table, '', file_name)

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
    def _save_as_csv(result_table, algorithm_name, file_name):
        base_dir = os.path.dirname(__file__)
        dir_path = os.path.join(base_dir, '../log/clustering_log/' + algorithm_name)
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
    tester = MainTester()

    # part1
    # tester.find_best_threshold(main.clustering2, 0.3, 0.8, 0.05, True, 1)
    # tester.find_best_threshold_with_ratio(main.clustering1, 0.9, 0.1, 0.4, 0.8, 0.05, True, 25)
    # tester.find_best_threshold_with_ratio(main.clustering3, 0.9, 0.1, 0.4, 0.8, 0.05, True, 100)
    # tester.find_best_threshold_with_ratio(main.clustering4, 0.6, 0.4, 0.4, 0.8, 0.05, True, 100)

    # part2
    # tester.find_best_ratio(main.clustering1, [(0.5, 0.6), (0.6, 0.6), (0.7, 0.6), (0.8, 0.55),
    #                                          (0.8, 0.6), (0.9, 0.55)], sampling=True, times=25)
    # tester.find_better_args_in_algorithm3(0.55, True, 20)
    # tester.find_best_ratio(main.clustering4, [(0.1, 0.6), (0.2, 0.6), (0.3, 0.55), (0.4, 0.55),
    #                                           (0.5, 0.55), (0.6, 0.55), (0.7, 0.55), (0.8, 0.55),
    #                                           (0.9, 0.55)], sampling=True, times=100)
    # part3
    # tester.find_best_ratio(main.clustering3, [(0.4, 0.45), (0.4, 0.5), (0.4, 0.55), (0.5, 0.6),
    #                                          (0.5, 0.65), (0.6, 0.65), (0.6, 0.7), (0.7, 0.75),
    #                                          (0.8, 0.8)], sampling=True, times=100)
    # part4
    # tester.compare_clustering(times=100)

    # tester.compare_clustering_using_real_data(start_month='2016/06', days=30)
    # tester.test_time_complexity()
    print('test finished in {0:.2f} seconds'.format(time.time() - start_time))
