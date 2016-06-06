import code.model.ptt_article_fetcher as fetcher
import jsonpickle
import os


def _get_cluster_from_topic_list(file_name="topics_list.txt"):
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
            days = int(end_day) - int(data[1])
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

            cluster = {'unique_titles': set([a.title for a in articles]), 'size': len(articles)}
            cluster['unique_size'] = len(cluster['unique_titles'])
            cluster['articles'] = articles
            cluster['keyword'] = keyword
            cluster['unique_ratio'] = round(int(cluster['unique_size']) / int(cluster['size']), 2)
            clusters.append(cluster)
    return clusters


def make_test_data(file_name="test_clusters.json"):
    clusters = _get_cluster_from_topic_list()
    check_duplicate_article(clusters)
    store_data(file_name, clusters)
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
    clusters = make_test_data()
    number_articles = len([a for cluster in clusters for a in cluster['articles']])
    print('\ncached {} clusters, {} articles'.format(len(clusters), number_articles))
