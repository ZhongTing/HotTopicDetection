from sklearn import metrics
from numpy import array as array
import code.test.make_test_data as test
from code.model.my_tokenize.tokenizer import cut
from code.model.keywords_extraction import keywords_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def _get_article_cluster_doc(clusters):
    _mark_cluster_number(clusters)
    labeded_articles = [article for cluster in clusters for article in cluster['articles']]
    labeded_articles.sort(key=lambda a: a.id)
    return [article.cluster_number for article in labeded_articles]


def _mark_cluster_number(clusters):
    for i in range(len(clusters)):
        for article in clusters[i]['articles']:
            article.cluster_number = i


def validate_clustering(cluster_ground_truth, cluster_predict):
    labels_true = _get_article_cluster_doc(cluster_ground_truth)
    labels_pred = _get_article_cluster_doc(cluster_predict)
    result = {
        'rand': '{0:.2f}'.format(metrics.adjusted_rand_score(labels_true, labels_pred)),
        'mutual_info': '{0:.2f}'.format(metrics.adjusted_mutual_info_score(labels_true, labels_pred)),
        'homogeneity': '{0:.2f}'.format(metrics.homogeneity_score(labels_true, labels_pred)),
        'completeness': '{0:.2f}'.format(metrics.completeness_score(labels_true, labels_pred)),
        'v_measure': '{0:.2f}'.format(metrics.v_measure_score(labels_true, labels_pred))
        # 'silhouette_index': '{0:.2f}'.format(silhouette_index(cluster_predict))
    }
    return result


def __split_string(article):
    tokens = cut(article.title)
    tokens.extend(keywords_extraction([article], 1))
    return ' '.join(tokens)


def silhouette_index(clusters):
    data = array([__split_string(a) for cluster in clusters for a in cluster['articles']])
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=200, min_df=2, stop_words='english')
    vectorizer = HashingVectorizer(stop_words='english', binary=False)

    X = vectorizer.fit_transform(data)
    svd = TruncatedSVD(15)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    try:
        # X = array([a.vector for cluster in clusters for a in cluster['articles']])
        labels = array([i for i in range(len(clusters)) for a in clusters[i]['articles']])
        score = metrics.silhouette_score(X, labels, metric='euclidean')
        return score
    except:
        return 0


if __name__ == '__main__':
    clusters = test.get_test_clusters()
    score = silhouette_index(clusters)
    print(score)
