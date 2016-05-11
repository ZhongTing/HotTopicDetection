from copy import deepcopy
import random


class Article:
    """docstring for Article"""

    def __init__(self, arg):
        # super(Article, self).__init__()
        self._title = arg

    @property
    def title(self):
        return self._title


def get_mock_aritcles(title_list):
    articles = []
    for title in title_list:
        articles.append(Article(title))
    return articles


def test_clusting(articles):
    clustings = []
    for article in articles:
        current_fit_clusting = None
        current_max_similarity = 0

        for clusting in clustings:
            compute_similarity = random.random()
            try:
                if compute_similarity > current_max_similarity:
                    current_max_similarity = compute_similarity
                    current_fit_clusting = clusting
            except ValueError:
                return

        if current_max_similarity < threshold:
            new_clusting = {'articles': [deepcopy(article)]}
            clustings.append(new_clusting)
        else:
            current_fit_clusting['articles'].append(deepcopy(article))

    print('-------------------')
    for i in range(len(clustings)):
        print('clusting ' + str(i))
        for aritlce in clustings[i]['articles']:
            print('bbb', aritlce.title)
            print('ddd', aritlce.title)
            t = article.title
            print('aaa', t)
            print('-----------')

    print('-------------------')
    for i in range(len(clustings)):
        print('clusting ' + str(i))
        for article in clustings[i]['articles']:
            print('bbb', article.title)
            print('ddd', article.title)
            t = article.title
            print('aaa', t)
            print('-----------')


debug_mode = False
threshold = 0.5
data = [
    "內閣總辭前槍決 法界：不想把這條大魚送",
    "「時間到了」…就得伏法　刑場槍決流程",
    "「時間到了」…就得伏法　刑場槍決流程",
    "鄭捷520前伏法? 馬：人命關天 不可能短時",
    "520前槍決鄭捷？　羅瑩雪：會開會討論",
    "鄭捷520前伏法? 馬英九：不可能短時間 人",
    "鄭捷520前伏法? 馬英九：不可能短時間 人",
    "鄭捷520前伏法？　馬英九：不可能短時間",
    "鄭捷520前伏法? 馬英九：不可能短時間 人",
    "鄭捷520前執行死刑？ 羅瑩雪：會開會討論",
    "520前執行死刑 羅瑩雪：不知道",
    "剩10天就要下台了,馬英九還是忘了一件事",
    "剩10天就要下台了,馬英九還是忘了一件事"
]
test_clusting(get_mock_aritcles(data))
