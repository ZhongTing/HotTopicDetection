from urllib import request
from urllib.parse import urlencode
import json
import re
import sys


class Article(object):
    """docstring for Article"""

    def __init__(self, arg):
        super(Article, self).__init__()
        if 'id' in arg:
            self.id = arg['id']
        if 'title' in arg:
            self.title = re.sub('.*?]', '', arg['title'][0])
        if 'author' in arg:
            self.author = arg['author'][0]
        if 'content' in arg:
            self.content = re.sub("-- ※ 發信站: 批踢踢實業坊\(ptt.cc.*", "",
                                  " ".join(arg['content'].split()))
            self.content = re.sub("https?:[\w#/\.=\?\&]*", "", self.content)
        if 'comments' in arg:
            self.comments = json.loads(arg['comments'])
            self.comments_content = []
            for comment in self.comments:
                self.comments_content.append(comment[2])

    def __repr__(self):
        return json.dumps(self.__dict__)


def fetch_articles(title, number=20, days=-1, page=1, only_title=False):
    server_url = 'http://140.124.183.7:8983/solr/HotTopicData/select?'
    post_args = {'q': 'title:*' + title + '*', 'rows': number, 'start': (page - 1) * number + 1}
    if days >= 0:
        post_args['fq'] = 'timestamp:[NOW/DAY-' + str(days) + 'DAYS TO NOW/DAY]'
    if only_title:
        post_args["fl"] = 'title'
    print(post_args)
    url = server_url + 'sort=timestamp+desc&wt=json&indent=true&' + urlencode(post_args)

    req = request.urlopen(url)
    encoding = req.headers.get_content_charset()
    sys_encoding = sys.stdin.encoding
    json_data = req.read().decode(encoding).encode(
        sys_encoding, 'replace').decode(sys_encoding)
    return parse_to_articles(json_data)


def parse_to_articles(json_data):
    articles = []
    for data in json.loads(json_data)['response']['docs']:
        articles.append((Article(data)))
    return articles
