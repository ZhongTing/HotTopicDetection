import gensim.models
from gensim import matutils
from numpy import array, dot, mean
import code.model.ptt_article_fetcher as fetcher
from code.model.my_tokenize.tokenizer import cut
import code.test.make_test_data as test_data
import code.model.lda as lda
import random
from code.clustering_validation import validate_clustering
import time


def get_test_clusters(sample_pick=False):
    origin_clusters = test_data.get_test_clusters()
    if sample_pick is False:
        return origin_clusters
    else:
        clusters = []
        for cluster in origin_clusters:
            pick_number = random.randint(1, len(cluster))
            sample = random.sample(cluster['articles'], pick_number)
            cluster['articles'] = sample
            clusters.append(cluster)

    return clusters


def get_test_articles(clusters=get_test_clusters()):
    articles = []
    for cluster in clusters:
        articles.extend(cluster['articles'])
    random.shuffle(articles)
    return articles


def get_ptt_articles():
    return fetcher.fetch_articles('*', number=3000, days=1)


def load_model():
    t = time.time()
    model_path = 'model/bin/model_82w.bin'
    model = gensim.models.Word2Vec.load(model_path)
    t = int(time.time() - t)
    print('spend {}s to load word2vec model from {}'.format(t, model_path))
    return model


def compute_vector(model, string, need_log=False):
    tokens = cut(string, using_stopwords=True, simplified_convert=True, log=need_log)
    if len(tokens) > 0 and (tokens[-1] in ['八卦', '卦']):
        del tokens[-1]
    if need_log is True:
        print(tokens)
    tokens_not_found = [word for word in tokens if word not in model]
    if len(tokens_not_found) is not 0:
        log('token not in model :' + " ".join(tokens_not_found))
    v1 = [model[word] for word in tokens if word in model]
    if len(v1) is 0:
        print('invalid article: \'' + string + '\'')
        return None
    vector = matutils.unitvec(array(v1, float).mean(axis=0))
    return vector


def compute_article_vector(model, articles):
    for article in articles:
        article.vector = compute_vector(model, article.title)


def initialize_clusters(articles):
    clusters = []
    for article in articles:
        if article.vector is None:
            print('initial error, invalid article: \'' + article.title + '\'')
            continue

        not_found = True
        for cluster in clusters:
            if cluster['articles'][-1].title == article.title:
                cluster['articles'].append(article)
                not_found = False
                break

        if not_found:
            clusters.append({'centroid': article.vector, 'articles': [article]})
    return clusters


def get_cluster_keyword(cluster):
    input_datas = [a.title + ' ' + a.content for a in cluster['articles']]
    model = lda.build_lda_model(input_datas, 1)
    return lda.get_topic(model, num_topics=1, num_words=5)[0]


def merge_clusters(clusters, threshold):
    clusters_after_merge = []
    while len(clusters) != 0:
        target_cluster = clusters.pop()
        highest_similarity = 0
        candidate_cluster = None
        for cluster in clusters:
            similarity = compute_similarily(cluster, target_cluster)
            if similarity > highest_similarity:
                highest_similarity = similarity
                candidate_cluster = cluster
        if highest_similarity > threshold:
            log('merged: cluster {} with {}'.format(
                candidate_cluster['articles'][0].title, target_cluster['articles'][0].title))
            combined(candidate_cluster, target_cluster)
        else:
            clusters_after_merge.append(target_cluster)
    return clusters_after_merge


def combined(cluster, merged_cluster):
    cluster['articles'].extend(merged_cluster['articles'])
    cluster['centroid'] = sum([a.vector for a in cluster['articles']]) / len(cluster['articles'])


def compute_similarily(cluster_a, cluster_b):
    return dot(cluster_a['centroid'], cluster_b['centroid'])


def print_clusters(clusters, print_title=False):
    for i in range(len(clusters)):
        cluster = clusters[i]
        score = sum([article.score for article in cluster['articles']])
        print('cluster', i, 'score', score, 'amount', len(cluster['articles']))
        print(get_cluster_keyword(cluster))
        if print_title is True:
            for article in cluster['articles']:
                print(article.title)
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


def find_closest_cluster(clusters):
    max_similarity = 0
    cluster_pair = None
    if len(clusters) <= 2:
        return (clusters[0], clusters[-1])

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            similarity = compute_similarily(clusters[i], clusters[j])
            if similarity > max_similarity:
                max_similarity = similarity
                cluster_pair = (clusters[i], clusters[j])
    return cluster_pair


def clustering1(articles, threshold):
    clusters = initialize_clusters(articles)
    cluster_pair = find_closest_cluster(clusters)
    while cluster_pair[0] is not cluster_pair[1] and compute_similarily(cluster_pair[0], cluster_pair[1]) > threshold:
        combined(cluster_pair[0], cluster_pair[1])
        try:
            clusters.remove(cluster_pair[1])
        except ValueError:
            print(cluster_pair[0])
            print(cluster_pair[1])
            print(clusters.index(cluster_pair[1]).all())
            return []
        cluster_pair = find_closest_cluster(clusters)
        print(len(clusters))

    return clusters


def clustering2(articles, threshold):
    clusters = initialize_clusters(articles)
    return merge_clusters(clusters, threshold)


def clustering(algorithm, threshold, articles):
    if algorithm is 1:
        clusters = clustering1(articles, threshold)
    elif algorithm is 2:
        clusters = clustering2(articles, threshold)
    return clusters


def test_clustering(algorithm, threshold, model=None, labeled_clusters=None, articles=None):
    model = load_model()
    labeled_clusters = get_test_clusters()
    articles = get_test_articles(labeled_clusters)
    compute_article_vector(model, articles)
    clusters = clustering(algorithm, threshold, articles)
    print_clustering_info(clusters, articles)
    print_validation_result(labeled_clusters, clusters)


def find_best_threshold(model, algorithm, random, start_th=0.2, increase_times=5, increase_count=0.1, test_times=1):

    test_list = []
    for i in range(test_times):
        labeled_clusters = get_test_clusters(random)
        articles = get_test_articles(labeled_clusters)
        compute_article_vector(model, articles)
        threshold = start_th
        result_list = []
        for i in range(increase_times):
            clusters = clustering(algorithm, threshold, articles)
            result = validate_clustering(labeled_clusters, clusters)
            result_list.append({'threshold': '{0:.2f}'.format(threshold), 'result': result})
            threshold += increase_count
        test_list.append({'result': result_list})

    print('algorithm ', algorithm)
    score_table = {}
    for test in test_list:
        result_list = test['result']
        for result_item in result_list:
            result = result_item['result']
            threshold = result_item['threshold']
            if threshold not in score_table:
                score_table[threshold] = {'v': [float(result['v_measure_score'])],
                                          'rand': [float(result['adjusted_rand_score'])],
                                          'mi': [float(result['adjusted_mutual_info_score'])]}
            else:
                score_table[threshold]['v'].append(float(result['v_measure_score']))
                score_table[threshold]['rand'].append(float(result['adjusted_rand_score']))
                score_table[threshold]['mi'].append(float(result['adjusted_mutual_info_score']))

            # print(threshold, result)

        test['score'] = {
            'v': max([(i['result']['v_measure_score'], i['threshold']) for i in result_list]),
            'rand': max([(i['result']['adjusted_rand_score'], i['threshold']) for i in result_list]),
            'mi': max([(i['result']['adjusted_mutual_info_score'], i['threshold']) for i in result_list])
        }
        print(test['score'])

    average_score = {
        'v': max([(mean(score_table[threshold]['v']), threshold) for threshold in score_table]),
        'rand': max([(mean(score_table[threshold]['rand']), threshold) for threshold in score_table]),
        'mi': max([(mean(score_table[threshold]['mi']), threshold) for threshold in score_table])
    }
    for key in average_score:
        print(key, '({0:.2f}, {1})'.format(average_score[key][0], average_score[key][1]))


def main():
    model = load_model()
    articles = get_ptt_articles()
    compute_article_vector(model, articles)
    clusters = clustering(2, 0.45, articles)
    print_clustering_info(clusters, articles)
    clusters = sorted(clusters, key=lambda cluster: sum([a.score for a in cluster['articles']]), reverse=True)
    print_clusters(clusters[0:5], True)


def log(string):
    if debug_mode is True:
        print(string)


debug_mode = False
# model = load_model()
# test_clustering(algorithm=2, threshold=0.45, model=model):
# simulate('20160509_2000_remove八卦', cluster_number=119)
# find_best_threshold(model, 1, 0.15, 8, 0.05)
# find_best_threshold(model, 2, False, 0.35, 3, 0.05, 3)
# find_best_threshold(model, 2, True, 0.35, 3, 0.05, 5)

main()
