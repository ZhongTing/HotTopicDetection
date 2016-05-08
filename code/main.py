from ptt_article_fetcher import fetch_articles, Article
import gensim.models
from gensim import matutils
from tokenizer import cut
from numpy import array, dot
import random


def get_test_articles():
    all_clusting = [
        # fetch_articles("人生勝利組", number=5),
        # fetch_articles("死刑", number=5),
        fetch_articles("*", number=2000)
    ]
    articles = []
    for clusting in all_clusting:
        articles.extend(clusting)
    random.shuffle(articles)
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


def compute_vector(model, str):
    tokens = cut(str, using_stopword=False, simplified_convert=True)
    tokens_not_found = [word for word in tokens if word not in model]
    if len(tokens_not_found) is not 0:
        print('token not in model :' + " ".join(tokens_not_found))
    v1 = [model[word] for word in tokens if word in model]
    if len(v1) is 0:
        raise Exception()
    vector = matutils.unitvec(array(v1, float).mean(axis=0))
    return vector


def compute_article_vector(model, articles):
    for article in articles:
        try:
            article.vector = compute_vector(model, article.title)
        except Exception:
            print('invalid article:')
            print(article)
            article.vector = None


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

        if current_max_similarity < 0.45:
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
            current_fit_clusting['articles'].append(article)

    print('-------------------')
    for i in range(len(clustings)):
        if len(clustings[i]['articles']) < 2:
            continue
        print('clusting ' + str(i))
        for aritlce in clustings[i]['articles']:
            print(aritlce)


def log(str):
    if debug_mode is True:
        print(str)


def validate(title, another_title):
    model = load_model()
    vector1 = compute_vector(model, title)
    vector2 = compute_vector(model, another_title)
    similarity = dot(vector1, vector2)
    print(similarity)

# test_vector_centroid_similarity()
debug_mode = True
data = [
    "有沒有成為人生勝利組最簡易的SOP?",
    "又死了兩個孩童，怎麼不趕快通過唯一死刑",
    "怎麼樣就算是人生勝利組？",
    "死刑定讞之後為什麼遲遲不槍決",
    "娶到周曉菁算不算人生勝利組?",
    "40多歲才考上公務員算人生勝利組嗎?",
    "有沒有法官不愛判死刑的八卦 ???",
    "娶到周曉菁算不算人生勝利組?",
    "柯震東是人生勝利組嗎?",
    "現在還不到台灣自決建國的時機?",
    "有沒有成為人生勝利組最簡易的SOP?",
    "死刑定讞之後為什麼遲遲不槍決",
    "死刑定讞之後為什麼遲遲不槍決",
    "四叉貓算不算肥宅中的人生勝利組",
    "為啥美國有死刑可以不用鳥兩公約",
    "有沒有成為人生勝利組最簡易的SOP?",
    "有沒有成為人生勝利組最簡易的SOP?",
    "參與死刑判決 日平民裁判官心理障礙",
    "求學受教育的期間為何這麼長?",
    "不判決死刑是因為總量管制的原因嗎？",
    "還有哪些罪應該判死刑",
    "連殺8雇主不手軟　廣東恐怖保母遭判死刑",
    "他在信中寫上這3個字　慘遭主管洗臉",
    "蘋果將推iPad Air 3？陸媒曝光規格"
]
# test_clusting(get_mock_aritcles(data))
test_clusting()
# validate('美研社消失的卦', '有出殯音樂的八卦嗎?')
