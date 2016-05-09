from ptt_article_fetcher import fetch_articles, Article
import gensim.models
from gensim import matutils
from tokenizer import cut
from numpy import array, dot
import re
from copy import deepcopy
# import random


def get_test_articles():
    all_clusting = [
        # fetch_articles("人生勝利組", number=5),
        # fetch_articles("死刑", number=5),
        fetch_articles("*", number=2000)
    ]
    articles = []
    for clusting in all_clusting:
        articles.extend(clusting)
    # random.shuffle(articles)
    return articles


def get_mock_aritcles(title_list):
    articles = []
    for title in title_list:
        articles.append(Article({'title': [title]}))
    return articles


def load_model():
    model_path = 'bin/model_5_78w.bin'
    model = gensim.models.Word2Vec.load(model_path)
    print('load word2vec model from ' + model_path)
    return model


def compute_vector(model, str, log=False):
    tokens = cut(str, using_stopword=True, simplified_convert=True, log=log)
    if len(tokens) > 0 and (tokens[-1] in ['八卦', '卦']):
        del tokens[-1]
    if log is True:
        print(tokens)
    tokens_not_found = [word for word in tokens if word not in model]
    if len(tokens_not_found) is not 0:
        print('token not in model :' + " ".join(tokens_not_found))
    v1 = [model[word] for word in tokens if word in model]
    if len(v1) is 0:
        print('invalid article: \'' + str + '\'')
        return None
    vector = matutils.unitvec(array(v1, float).mean(axis=0))
    return vector


def compute_article_vector(model, articles):
    for article in articles:
        article.vector = compute_vector(model, article.title)


def test_vector_centroid_similarity():

    model = load_model()
    articles = get_test_articles()
    compute_article_vector(model, articles)

    vector_stack = []
    for i in range(0, len(articles)):
        article = articles[i]
        vector = article.vector
        vector_stack.append(vector)
    if len(vector_stack) > 1:
        print(article.title)
        print(articles[i - 1].title)
        si = dot(vector_stack[-1], vector_stack[-2])
        print('previos similarity = ' + str(si))
        # mean
        si2 = dot(array(vector_stack[0:-1]).mean(axis=0), vector_stack[-1])
        print('mean similarity = ' + str(si2))


def test_clusting(articles=None):
    model = load_model()
    if articles is None:
        articles = get_test_articles()
    compute_article_vector(model, articles)
    clustings = []
    for article in articles:
        current_fit_clusting = None
        current_max_similarity = 0

        if article.vector is None:
            print('===============fuck')
            continue
        for clusting in clustings:
            compute_similarity = dot(article.vector, clusting['centroid'])
            try:
                if compute_similarity > current_max_similarity:
                    current_max_similarity = compute_similarity
                    current_fit_clusting = clusting
            except ValueError:
                print(article)
                compute_vector(model, article.title, True)
                print(ValueError)
                print(article.vector)
                return

        if current_max_similarity < threshold:
            log('new clusting ' + article.title)
            if current_fit_clusting is not None:
                log('compare with clusting ' + current_fit_clusting['articles'][0].title)
                log('with max similarity ' + str(current_max_similarity) + "\n")
            new_clusting = {'centroid': article.vector, 'articles': [article]}
            clustings.append(new_clusting)
        else:
            size = len(current_fit_clusting['articles'])
            current_fit_clusting['centroid'] = (
                current_fit_clusting['centroid'] * size + article.vector) / (size + 1)
            current_fit_clusting['articles'].append(deepcopy(article))

    print('-------------------')
    current_max_clusting_size = 0
    min_clusting_count = 0
    min_clusting_size = 2
    # clusting_with_same_title = 0
    for i in range(len(clustings)):
        size = len(clustings[i]['articles'])
        if size > current_max_clusting_size:
            current_max_clusting_size = size
        if size < min_clusting_size:
            min_clusting_count += 1
            continue
        # if len(set([article.title for aritcle in clustings[i]['articles']])) < 2:
        #     print(set([article.title for aritcle in clustings[i]['articles']]))
        #     clusting_with_same_title += 1
        #     continue

        print('clusting ' + str(i))
        for aritlce in clustings[i]['articles']:
            print(aritlce.title)

    print('total articles : ', len(articles))
    print('unrepeat titles : ', len(set([article.title for article in articles])))
    print('total clustings : ', len(clustings))
    # print('clustings with same title ', clusting_with_same_title)
    # print('clustings with different title ', len(clustings) - clusting_with_same_title)
    print('max_clusting_size : ', current_max_clusting_size)
    print('clustings size under {} : {}'.format(min_clusting_size, min_clusting_count))


def log(str):
    if debug_mode is True:
        print(str)


def validate(data):
    model = load_model()
    for i in range(1, len(data)):
        print(data[i - 1])
        vector1 = compute_vector(model, data[i - 1], True)
        print(data[i])
        vector2 = compute_vector(model, data[i], True)
        if vector1 is None or vector2 is None:
            print('vector error')
            continue
        similarity = dot(vector1, vector2)
        print(similarity)


def simulate(file_name, clusting_number):
    with open('word2vec_log/clusting_log/' + file_name, 'r', encoding='utf8') as file:
        content = file.read()
        pattern = re.compile('clusting ' + str(clusting_number) + '\n([\W\w]*?)\nclusting')
        titles = re.findall(pattern, content)[0].split('\n')
        validate(titles)


debug_mode = True
threshold = 0.55
# test_vector_centroid_similarity()
# test_clusting(get_mock_aritcles(data))
test_clusting()
# simulate('20160509_2000_remove八卦', clusting_number=119)
