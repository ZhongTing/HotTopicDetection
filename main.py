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
        self.content = re.sub("https?:[\w#/\.=\?\&]*", "", self.content)
        self.comments = json.loads(arg['comments'])
        self.comments_contnet = []
        for comment in self.comments:
            self.comments_contnet.append(comment[2])

    def __repr__(self):
        return json.dumps(self.__dict__)


def fetchArticle(title, number=20):
    serverUrl = 'http://140.124.183.7:8983/solr/HotTopicData/select?'
    url = serverUrl + 'sort=timestamp+desc&wt=json&indent=true&' + \
        urlencode({'q': 'title:*' + title + '*', 'rows': number})

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


def log(file, object):
    print(object)
    print(object, end="\n", file=file)


def buildLdaBykeyword(keyowords, num_article_for_search):
    num_topics = len(keyowords)

    filename = 'data/' + str("_".join(keyowords) + "_" + str(num_article_for_search))
    print(filename)
    file = open(filename, 'w', encoding="utf-8")
    articles = []
    for keyoword in keyowords:
        article_json = fetchArticle(keyoword, num_article_for_search)
        articles_keyword = parseToArticle(article_json)
        articles += articles_keyword
        log(file, "%s : %d" % (keyoword, len(articles_keyword)))

    with open('stopword.json', encoding='utf-8') as data_file:
        stopwords = json.load(data_file)

    texts = []
    for article in articles:
        tokens = jieba.cut_for_search(article.title + article.content)
        tokens = [i for i in list(tokens) if i not in stopwords]
        texts.append(tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    start = time.time()
    ldamodel = models.ldamodel.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=1)

    pattern = re.compile('\*(.*?) ')
    for topic in ldamodel.show_topics(num_topics=num_topics):
        log(file, pattern.findall(topic[1]))
    end = time.time()
    log(file, "model train time : " + str(end - start))

    print("\n\n\n\n", file=file)
    for article in articles:
        print(article.title, end="\n", file=file)

    file.close()


def test1():
    keyowords_candidate = ['隨機', '巴拿馬', '慈濟']
    key_index_set = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    for num_topics in [10, 20, 50, 100]:
        for key_index in key_index_set:
            keyowords = []
            for index in key_index:
                keyowords.append(keyowords_candidate[index])
            buildLdaBykeyword(keyowords, num_topics)


def test2():
    keyowords = ['王建民', '柯文哲', '和田光司', '義美', '統神']
    # keyowords = ['邊緣人']
    buildLdaBykeyword(keyowords, 9)


def test3(keyoword, number):
    buildLdaBykeyword([keyoword], number)


# test2()
#test3("林鳳營", 200)

buildLdaBykeyword(["慈濟", "王建民"], 100)
