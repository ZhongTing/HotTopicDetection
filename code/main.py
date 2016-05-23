import gensim.models
from gensim import matutils
from numpy import array, dot, mean
import code.model.ptt_article_fetcher as fetcher
from code.model.my_tokenize.tokenizer import cut
import code.test.make_test_data as test_data
import random
from code.clustering_validation import validate_clustering, silhouette_index
import time
import os
from code.model.keywords_extraction import keywords_extraction


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
    return fetcher.fetch_articles('*', number=2000, days=1)


def load_model(model_path='model/bin/ngram_300_3_83w.bin'):
    t = time.time()
    model = gensim.models.Word2Vec.load(model_path)
    t = int(time.time() - t)
    print('spend {}s to load word2vec model from {}'.format(t, model_path))
    return model


def compute_vector(model, input_data, need_log=False):
    if isinstance(input_data, str):
        tokens = cut(input_data, using_stopwords=True, simplified_convert=True, log=need_log)
    else:
        tokens = input_data
    if len(tokens) > 0 and (tokens[-1] in ['八卦', '卦']):
        del tokens[-1]
    if need_log is True:
        print(tokens)
    tokens_not_found = [word for word in tokens if word not in model]
    if len(tokens_not_found) is not 0:
        log('token not in model :' + " ".join(tokens_not_found))
    v1 = [model[word] for word in tokens if word in model]
    if len(v1) is 0:
        print('invalid article: \'' + input_data + '\'')
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
    return [
        keywords_extraction(cluster['articles'], 0), 
        keywords_extraction(cluster['articles'], 1)
    ]


def merge_clusters(model, clusters, threshold, combined_method=0, similarity_method=0):
    clusters_after_merge = []
    while len(clusters) != 0:
        target_cluster = clusters.pop()
        highest_similarity = 0
        candidate_cluster = None
        for cluster in clusters:
            similarity = compute_similarily(cluster, target_cluster, similarity_method)
            if similarity > highest_similarity:
                highest_similarity = similarity
                candidate_cluster = cluster
        if highest_similarity > threshold:
            log('merged: cluster {} with {}'.format(
                candidate_cluster['articles'][0].title, target_cluster['articles'][0].title))
            combined(model, candidate_cluster, target_cluster, combined_method)
        else:
            clusters_after_merge.append(target_cluster)
    return clusters_after_merge


def combined(model, cluster, merged_cluster, combined_method):
    cluster['articles'].extend(merged_cluster['articles'])
    if combined_method is 0:
        cluster['centroid'] = sum([a.vector for a in cluster['articles']]) / len(cluster['articles'])
    elif combined_method is 1:
        cluster['centroid'] = sum([a.vector for a in cluster['articles']]) / len(cluster['articles'])
        cluster['keywords'] = compute_vector(model, keywords_extraction(cluster['articles']))


def compute_similarily(cluster_a, cluster_b, similarity_method):
    if similarity_method is 0:
        return dot(cluster_a['centroid'], cluster_b['centroid'])
    elif similarity_method is 1:
        return min([dot(a.vector, b.vector) for a in cluster_a['articles'] for b in cluster_b['articles']])
    elif similarity_method is 2:
        title_score = dot(cluster_a['centroid'], cluster_b['centroid'])
        keywords_score = dot(cluster_a['keywords'], cluster_b['keywords'])
        return title_score * 0.8 + keywords_score * 0.2


def print_clusters(clusters, print_title=False):
    for i in range(len(clusters)):
        cluster = clusters[i]
        score = sum([article.score for article in cluster['articles']])
        print('cluster', i, 'score', score, 'amount', len(cluster['articles']))
        for keywords in get_cluster_keyword(cluster):
            print(keywords)
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


def clustering1(model, articles, threshold):
    clusters = initialize_clusters(articles)
    for cluster in clusters:
        cluster['keywords'] = compute_vector(model, keywords_extraction(cluster['articles']))
    return merge_clusters(model, clusters, threshold, combined_method=1, similarity_method=2)


def clustering2(model, articles, threshold):
    clusters = initialize_clusters(articles)
    return merge_clusters(model, clusters, threshold)


def clustering3(model, articles, threshold):
    clusters = initialize_clusters(articles)
    clusters = merge_clusters(model, clusters, 0.55)
    for cluster in clusters:
        cluster['centroid'] = compute_vector(model, keywords_extraction(cluster['articles']))
    clusters = merge_clusters(model, clusters, threshold, combined_method=0)
    return clusters


def clustering(model, algorithm, threshold, articles):
    if algorithm is 1:
        clusters = clustering1(model, articles, threshold)
    elif algorithm is 2:
        clusters = clustering2(model, articles, threshold)
    elif algorithm is 3:
        clusters = clustering3(model, articles, threshold)
    return clusters


def test_clustering(algorithm, threshold, model=None, labeled_clusters=None, articles=None):
    model = load_model()
    labeled_clusters = get_test_clusters()
    articles = get_test_articles(labeled_clusters)
    compute_article_vector(model, articles)
    clusters = clustering(model, algorithm, threshold, articles)
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
            clusters = clustering(model, algorithm, threshold, articles)
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
                score_table[threshold] = {}
            for key in sorted(result.keys()):
                if key not in score_table[threshold]:
                    score_table[threshold][key] = []
                score_table[threshold][key].append(float(result[key]))

            print(threshold, result)

        test['score'] = {}
        for key in sorted(result_list[0]['result'].keys()):
            test['score'][key] = max([(i['result'][key], i['threshold']) for i in result_list])

        print(test['score'])

    average_score = {}
    for key in sorted(score_table[threshold].keys()):
        average_score[key] = max([(mean(score_table[threshold][key]), threshold) for threshold in score_table])

    result = {}
    for key in sorted(average_score.keys()):
        result[key] = '({0:.2f}, {1})'.format(average_score[key][0], average_score[key][1])
    return result


def find_best_model():
    dir_path = 'model/bin'
    dirs = os.listdir(dir_path)
    model_path_list = [path for path in dirs if path[-3:] == 'bin']
    for model_path in model_path_list:
        model = load_model(os.path.join(dir_path, model_path))
        result = find_best_threshold(model, 2, False, 0.3, 5, 0.05, 3)
        print('using model', model_path.ljust(25), result)


def test_model(model, algorithm, threshold, random, times):
    result = find_best_threshold(model, algorithm, random, start_th=threshold, increase_times=1, test_times=times)
    print(result)


def main(algorithm, threshold=0.55):
    model = load_model()
    articles = get_ptt_articles()
    compute_article_vector(model, articles)
    clusters = clustering(model, algorithm, threshold, articles)
    print_clustering_info(clusters, articles)
    clusters = sorted(clusters, key=lambda cluster: sum([a.score for a in cluster['articles']]), reverse=True)
    print_clusters(clusters[0:5], True)
    print('silhouette_index', silhouette_index(clusters))


def log(string):
    if debug_mode is True:
        print(string)


debug_mode = False
model = load_model()
# test_clustering(algorithm=2, threshold=0.45, model=model):
# simulate('20160509_2000_remove八卦', cluster_number=119)
# print(find_best_threshold(model, 1, False, 0.5, 5, 0.05, 1))
# print(find_best_threshold(model, 2, False, 0.5, 9, 0.05, 1))
print(find_best_threshold(model, 3, False, 0.5, 9, 0.05, 1))

# main(1, 0.55)

# find_best_model()
# test_model(model, 2, 0.55, False, 3)

# clusters = get_test_clusters()
# for cluster in clusters:
#     compute_article_vector(model, cluster['articles'])
# score = silhouette_index(clusters)
# print(score)
