from urllib import request
from urllib.parse import urlencode
import json
import re
import sys
import time


class Article(object):
    """docstring for Article"""

    def __init__(self, arg):
        super(Article, self).__init__()
        self.push_score = 0
        self.boo_score = 0
        self.score = 0

        if 'id' in arg:
            self.id = arg['id']
        if 'title' in arg:
            self.title = re.sub('.*?]', '', arg['title'][0])
            self.title = re.sub('R:', '', self.title).strip()
        if 'author' in arg:
            self.author = arg['author'][0]
        if 'content' in arg:
            temp_content = re.sub("-- *\n※ 發信站: 批踢踢實業坊\(ptt.cc(\n|.)*", "", arg['content'])
            temp_content = re.sub("※ 引述.*\n", '', temp_content)
            temp_content = re.sub("(:.*\n)*", '', temp_content)
            temp_content = re.sub("https?:[\w\.=#/\?\&]*", "", temp_content)
            # 濾空行
            temp_content = re.sub(r'^ *\n', '', temp_content)
            # 幫不以標點符號而用enter換行加上句號
            temp_content = re.sub("(^[^，。？：！!?,\.\n:]+ *\n)", r'\1。', temp_content)
            # 把因字數太長的斷句的句子接在同一句
            temp_content = re.sub("([^，。、？：！!?,\. \n:]) *\n([^ \n])", r"\1\2", temp_content)
            # 把同一句標點符號再斷句
            temp_content = re.sub('([，。？：！!?,]+) *', r'\1\n', temp_content)
            self.content_sentence = [
                i for i in re.findall(r'([^ \n].+) *', temp_content) if i not in ['']]

            self.content = '\n'.join(self.content_sentence)
        if 'comments' in arg:
            self.comments = json.loads(arg['comments'])
            self.comments_content = []
            for comment in self.comments:
                if comment[0] == '推':
                    self.push_score += 1
                elif comment[0] == '噓':
                    self.boo_score += 1

            self.score = self.push_score - self.boo_score

    def __repr__(self):
        return self.title

    def getKey(self):
        return self.title

    def info(self):
        return 'article {}(推/噓/總):{}/{}/{}'.format(self.title, self.push_score, self.boo_score, self.score)


def fetch_articles(title, number=20, end_day='NOW/DAY', days=-1, page=1, only_title=False, fl=None, desc=True):
    start_time = time.time()
    server_url = 'http://140.124.183.7:8983/solr/HotTopicData/select?'
    post_args = {'q': 'title:*' + title + '*', 'rows': number, 'start': (page - 1) * number + 1}
    if days >= 0:
        if re.compile(r'\d{4}/\d{2}/\d{2}').match(end_day):
            start_day = "-".join(end_day.split('/')) + 'T00:00:00Z'
            end_day = "-".join(end_day.split('/')) + 'T23:59:59Z'
        post_args['fq'] = 'timestamp:[{}-{}DAYS TO {}]'.format(start_day, days, end_day)
    if only_title:
        post_args["fl"] = 'title'
    if fl:
        post_args["fl"] = fl
    print(post_args)

    desc_arg = 'sort=timestamp+desc&' if desc else ''
    url = server_url + desc_arg + 'wt=json&indent=true&' + urlencode(post_args)

    articles = []
    retry_times = 3
    while len(articles) is 0 and retry_times > 0:
        req = request.urlopen(url)
        encoding = req.headers.get_content_charset()
        sys_encoding = sys.stdin.encoding
        json_data = req.read().decode(encoding).encode(
            sys_encoding, 'replace').decode(sys_encoding)
        waiting_time = time.time() - start_time
        articles = parse_to_articles(json_data)
        print('fetch ' + str(len(articles)) + ' articles spend ' + str(waiting_time))
        if len(articles) is 0:
            print('error occur... retry fetching articles')
        retry_times -= 1

    return articles


def parse_to_articles(json_data):
    articles = []
    for data in json.loads(json_data)['response']['docs']:
        articles.append((Article(data)))
    return articles
