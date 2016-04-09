from urllib import request
from urllib.parse import urlencode
import json
import sys
import jieba
import re
import time
from gensim import corpora, models


class Article(object):
    """docstring for Article"""

    def __init__(self, arg):
        super(Article, self).__init__()
        self.id = arg['id']
        self.title = re.sub('.*?]', '', arg['title'][0])
        self.author = arg['author'][0]
        self.content = re.sub("-- ※ 發信站: 批踢踢實業坊\(ptt.cc.*", "",
                              " ".join(arg['content'].split()))
        self.comments = arg['comments']

    def __repr__(self):
        return json.dumps(self.__dict__)


def fetchArticle(title):
    serverUrl = 'http://140.124.183.7:8983/solr/HotTopicData/select?'
    url = serverUrl + 'sort=timestamp+desc&wt=json&indent=true&' + \
        urlencode({'q': 'title:*' + title + '*', 'rows': 50})
    print((url))

    req = request.urlopen(url)
    encoding = req.headers.get_content_charset()
    sys_encdoing = sys.stdin.encoding
    json_data = req.read().decode(encoding).encode(sys_encdoing, 'replace').decode(sys_encdoing)
    return json_data


def parseToArticle(json_data):
    articles = []
    for data in json.loads(json_data)['response']['docs']:
        articles.append((Article(data)))
    return articles


def main():
    num_topics = 3
    articles = parseToArticle(fetchArticle('隨機*人'))
    articles += parseToArticle(fetchArticle('巴拿馬'))
    articles += parseToArticle(fetchArticle('慈濟'))
    with open('stopword.json', encoding='utf-8') as data_file:
        stopwords = json.load(data_file)

    texts = []
    for article in articles:
        print(article.title)
        # print(article.content)
        tokens = jieba.cut(article.title + article.content, cut_all=False)
        tokens = [i for i in list(tokens) if i not in stopwords]
        texts.append(tokens)

    print("build dictionary")
    dictionary = corpora.Dictionary(texts)
    print("build corpus")
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("build LdaModel")
    start = time.time()
    ldamodel = models.ldamodel.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=20)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=5))
    end = time.time()
    print(end - start)

main()
