from sklearn import metrics


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
        'adjusted_rand_score': metrics.adjusted_rand_score(labels_true, labels_pred),
        'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(labels_true, labels_pred),
        'homogeneity_score': metrics.homogeneity_score(labels_true, labels_pred),
        'completeness_score': metrics.completeness_score(labels_true, labels_pred),
        'v_measure_score': metrics.v_measure_score(labels_true, labels_pred)
    }
    return result
