from urllib import request
from urllib.parse import urlencode
import json
import sys


class Article(object):
    """docstring for Article"""

    def __init__(self, arg):
        super(Article, self).__init__()
        self.id = arg['id']
        self.title = arg['title'][0]
        self.author = arg['author'][0]
        self.content = arg['content']
        self.comments = arg['comments']

    def __repr__(self):
        return json.dumps(self.__dict__)


def fetchArticle(title):
    serverUrl = 'http://140.124.183.7:8983/solr/HotTopicData/select?'
    url = serverUrl + 'sort=timestamp+desc&wt=json&indent=true&' + \
        urlencode({'q': 'title:*' + title + '*', 'rows': 20})
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
    articles = parseToArticle(fetchArticle('隨機*人'))
    for article in articles:
        print(article.title)

main()
