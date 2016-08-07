import python_code.model.ptt_article_fetcher as fetcher
import jsonpickle
import os
import re
import python_code.main as main
import random


def _get_cluster_from_topic_list(file_name="source/topics_list.txt"):
    path = get_path(file_name)
    clusters = []
    with open(path, mode='r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            if len(data) < 3:
                continue
            keyword = data[0]
            end_day = data[2]
            if not end_day.isdigit():
                end_day = data[1]
            # days = int(end_day) - int(data[1])
            end_day = '201{}/{}/{}'.format(end_day[0], end_day[1:3], end_day[3:5])
            # articles = fetcher.fetch_articles(keyword, number=200, end_day=end_day, days=days)
            start_day = data[1]
            start_day = '201{}/{}/{}'.format(start_day[0], start_day[1:3], start_day[3:5])
            articles = fetcher.fetch_articles_by_day_interval(keyword, number=200, start_day=start_day, end_day=end_day)
            if len(data) >= 4:
                negative = data[3]
                if negative[0] != '(':
                    negative = negative.split(',')
                    remove_negative_articles = []
                    for article in articles:
                        ok = True
                        for n in negative:
                            if n in article.title:
                                ok = False
                                break
                        if ok is True:
                            remove_negative_articles.append(article)
                    articles = remove_negative_articles

            cluster = _create_cluster(articles, keyword)
            clusters.append(cluster)
    return clusters


def _create_cluster(articles, keyword):
    if articles is None:
        return None
    cluster = {}
    cluster['unique_titles'] = set([a.title for a in articles])
    cluster['size'] = len(articles)
    cluster['unique_size'] = len(cluster['unique_titles'])
    cluster['articles'] = articles
    cluster['keyword'] = keyword
    cluster['unique_ratio'] = round(int(cluster['unique_size']) / int(cluster['size']), 2)
    return cluster


def _parse_cluster(path, one_article_per_cluster=False):
    result = []
    with open(path, mode='r', encoding='utf8') as f:
        lines = f.readlines()
        content = ''.join(lines)
        for cluster_string in re.split('\n\n', content):
            pattern = '\w{56}'
            id_list = re.findall(pattern, cluster_string)
            print(id_list)
            articles = fetcher.fetch_articles_with_id(id_list)
            if len(articles) != 0:
                if one_article_per_cluster is True:
                    for article in articles:
                        result.append((_create_cluster([article], None)))
                else:
                    result.append(_create_cluster(articles, None))
    return result


def make_test_data_from_label_data(input_folder):
    store_file_name = input_folder + '.json'
    clusters = []
    for file_name in ['clusters.txt', 'new_clusters.txt', 'noise.txt']:
        path = os.path.join('source', input_folder, file_name)
        clusters.extend(_parse_cluster(path, one_article_per_cluster=True if file_name is 'noise.txt' else False))
    check_duplicate_article(clusters)
    store_data(store_file_name, clusters)
    print(len(clusters))
    print(len([a for cluster in clusters for a in cluster['articles']]))


def sampling_test_data_into_source(data_file_name, sample_num=1000):
    clusters = get_test_clusters(data_file_name)
    target_id_list = [article.id for cluster in clusters for article in cluster['articles']]
    target_id_list = random.sample(target_id_list, sample_num)
    print('id list', len(target_id_list))
    print('origin cluster', len(clusters))
    print('origin articles', len([a for cluster in clusters for a in cluster['articles']]))

    remove_counter = 0
    for cluster in clusters:
        for article in cluster['articles']:
            for aid in target_id_list:
                if article.id is not aid:
                    cluster['articles'].remove(article)
                    remove_counter += 1
                    break
        if len(cluster['articles']) == 0:
            clusters.remove(cluster)
    print(len(clusters))
    print(len([a for cluster in clusters for a in cluster['articles']]))
    print('remove', remove_counter)
    file_clusters = open('clusters.txt', 'w', encoding='utf8')
    file_noise = open('noise.txt', 'w', encoding='utf8')
    file_new_clusters = open('new_clusters.txt', 'w', encoding='utf8')

    for cluster in clusters:
        title_set = set([article.title for article in cluster['articles']])
        if len(title_set) > 1:
            continue
        else:
            for article in cluster['articles']:
                if len(cluster['articles']) > 1:
                    print(article.id, article.title, file=file_new_clusters)
                else:
                    print(article.id, article.title, file=file_noise)
            if len(cluster['articles']) > 1:
                print('\n', file=file_new_clusters)
            clusters.remove(cluster)

    print(len(clusters))
    main.print_clusters(clusters, print_title=True, file=file_clusters)


def make_test_data_from_topic_list(output_name="test_clusters.json"):
    clusters = _get_cluster_from_topic_list()
    check_duplicate_article(clusters)
    store_data(output_name, clusters)
    return clusters


def check_duplicate_article(clusters):
    id_set = set()
    total_articles = []
    for cluster in clusters:
        total_articles.extend(cluster['articles'])
    for article in total_articles:
        if article.id not in id_set:
            id_set.add(article.id)
        else:
            print(article.id, article.title)


def get_test_clusters(file_name='test_clusters.json'):
    return get_data(file_name)


def store_one_day_data(day):
    obj = fetcher.fetch_articles('', 5000, end_day=day, days=1)
    day = day.replace('/', '', 3)
    store_data(file_name=day, data=obj)


def fetch_one_day_data(day):
    day = day.replace('/', '', 3)
    return get_data(day)


def store_data(file_name, data):
    path = get_path(folder_dir='test_data', file_name=file_name)
    with open(path, mode='w', encoding='utf8') as f:
        frozen = jsonpickle.encode(data)
        print(frozen, file=f)


def get_data(file_name):
    path = get_path(folder_dir='test_data', file_name=file_name)
    with open(path, mode='r', encoding='utf8') as f:
        json_string = f.read()
        data = jsonpickle.decode(json_string)
    return data


def get_path(file_name, folder_dir=None):
    base_dir = os.path.dirname(__file__)
    if folder_dir is None:
        folder_dir = base_dir
    else:
        folder_dir = os.path.join(base_dir, folder_dir)
        if os.path.exists(folder_dir) is False:
            os.mkdir(folder_dir)
    return os.path.join(folder_dir, file_name)


if __name__ == '__main__':
    # clusters = make_test_data_from_topic_list()
    # number_articles = len([a for cluster in clusters for a in cluster['articles']])
    # print('\ncached {} clusters, {} articles'.format(len(clusters), number_articles))
    make_test_data_from_label_data('20160629')
