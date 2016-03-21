from urllib import request
import json
import sys


class Article(object):
    """docstring for Article"""

    def __init__(self, arg):
        super(Article, self).__init__()
        self.title = arg['title']


def object_decoder(obj):
    if '__type__' in obj and obj['__type__'] == 'Article':
        return Article(obj["responsee"]["docs"])
    return obj

url = 'http://140.124.183.7:8983/solr/HotTopicData/select?indent=on&q=*:*&wt=json'
req = request.urlopen(url)
encoding = req.headers.get_content_charset()
sys_encdoing = sys.stdin.encoding
json_data = req.read().decode(encoding).encode(sys_encdoing, 'replace').decode(sys_encdoing)
print(json_data)
