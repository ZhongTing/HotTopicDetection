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
        self._labeled_clusters = test_data.get_test_clusters('20160615.json')

    def _get_test_articles(self):
        articles = [article for cluster in self._labeled_clusters for article in cluster['articles']]
        random.shuffle(articles)
        return articles

    def best_threshold(self, feature_extractor, linkage, similarity, start_th, end_th, step):
        file_name = 'best_threshold {} {} {}'.format(feature_extractor.args(), linkage, similarity)
        articles = self._get_test_articles()
        feature_extractor.fit(articles)
        result_table = {}
        threshold = start_th
        try:
            while threshold < end_th + step:
                clusters = HAC(threshold=threshold, linkage=linkage, similarity=similarity).fit(articles)
                result = validate_clustering(self._labeled_clusters, clusters)
                key = '{0:.2f}'.format(threshold)
                if key not in result_table:
                    result_table[key] = []
                result_table[key].append(result)
                threshold += step

            self._print_test_result(result_table)
            self._save_as_csv(result_table, feature_extractor.name(), file_name)
        except ValueError:
            return
        else:
            return

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
    tester = AgglomerativeClusteringTester()
    for only_title in [True, False]:
        feature_extractor = extractor.TFIDF(use_idf=True, only_title=only_title)
        for linkage in [HAC.LINKAGE_CENTROID, HAC.LINKAGE_COMPLETE, HAC.LINKAGE_SINGLE, HAC.LINKAGE_AVERAGE]:
            tester.best_threshold(feature_extractor, linkage, HAC.SIMILARITY_COSINE, 0.05, 0.4, step=0.05)


def title():
    tester = AgglomerativeClusteringTester()
    model = extractor.load_model('model/bin/ngram_300_3_83w.bin')
    feature_extractor = extractor.Title(model)
    for linkage in [HAC.LINKAGE_CENTROID, HAC.LINKAGE_COMPLETE, HAC.LINKAGE_SINGLE, HAC.LINKAGE_AVERAGE]:
        tester.best_threshold(feature_extractor, linkage, HAC.SIMILARITY_COSINE, 0.3, 0.9, step=0.05)


def extraction():
    tester = AgglomerativeClusteringTester()
    model = extractor.load_model('model/bin/ngram_300_3_83w.bin')
    feature_extractor = extractor.ContentExtraction(model, 0, 5, with_weight=True)
    for linkage in [HAC.LINKAGE_CENTROID, HAC.LINKAGE_COMPLETE, HAC.LINKAGE_SINGLE, HAC.LINKAGE_AVERAGE]:
        tester.best_threshold(feature_extractor, linkage, HAC.SIMILARITY_COSINE, 0.3, 0.9, step=0.05)

if __name__ == '__main__':
    start_time = time.time()
    # idf()
    # title()
    extraction()
    print('test finished in {0:.2f} seconds'.format(time.time() - start_time))
