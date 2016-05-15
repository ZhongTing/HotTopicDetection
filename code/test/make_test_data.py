import code.model.ptt_article_fetcher as fetcher
import jsonpickle
import os


def get_cluster_for_test(file_name="topics_list.txt"):
    path = get_path(file_name)
    clusters = []
    with open(path, mode='r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            keyword = data[0]
            end_day = data[2]
            if not end_day.isdigit():
                end_day = data[1]
            days = int(end_day) - int(data[1])
            end_day = '201{}/{}/{}'.format(end_day[0], end_day[1:3], end_day[3:5])
            articles = fetcher.fetch_articles(keyword, number=200, end_day=end_day, days=days)
            cluster = {'unique_titles': len(set([a.title for a in articles])), 'size': len(articles), }
            cluster['article'] = articles
            cluster['unique_ratio'] = round(int(cluster['unique_titles']) / int(cluster['size']), 2)
            clusters.append(cluster)
    return clusters


def make_test_data(file_name="test_clusters.json"):
    clusters = get_cluster_for_test()
    store_data(file_name, clusters)
    return clusters


def get_test_data(file_name='test_clusters.json'):
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


clusters = make_test_data()
# clusters = get_test_data()
for a in clusters:
    print(a['size'], a['unique_titles'], a['unique_ratio'])
