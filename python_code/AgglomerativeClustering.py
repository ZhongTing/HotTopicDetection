from numpy import dot


class AgglomerativeClustering:
    """docstring for AgglomerativeClustering"""

    def __init__(self, threshold):
        self.threshold = threshold
        pass

    def fit(self, articles):
        clusters = self._init_clusters(articles)
        print('building similarity table...')
        cluster_pair_list = self._build_cluster_pair_list(clusters)
        most_closest_pair = cluster_pair_list[0]
        while len(cluster_pair_list) > 0 and most_closest_pair['similarity'] > self.threshold:
            cluster_a = most_closest_pair['key']
            cluster_b = most_closest_pair['target']
            if len(cluster_pair_list) % 200 == 0:
                print(len(cluster_pair_list))
            self._merge_clusters(cluster_a, cluster_b, cluster_pair_list, clusters)
            most_closest_pair = cluster_pair_list[0]

        return clusters

    def quick_fit(self, articles):
        clusters_after_merge = []
        clusters = self._init_clusters(articles)
        while len(clusters) != 0:
            target_cluster = clusters.pop()
            highest_similarity = 0
            candidate_cluster = None
            for cluster in clusters:
                similarity = self._similarity(cluster, target_cluster)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    candidate_cluster = cluster
            if highest_similarity > self.threshold:
                self._merge_clusters(candidate_cluster, target_cluster)
            else:
                clusters_after_merge.append(target_cluster)
        return clusters_after_merge

    @staticmethod
    def _init_clusters(articles):
        clusters = []
        counter = 0
        for article in articles:
            if article.vector is None:
                raise ValueError("no articles vector")

            not_found = True
            for cluster in clusters:
                if dot(article.vector, cluster['vector'] == 1.0):
                    cluster['articles'].append(article)
                    not_found = False
                    break

            if not_found:
                clusters.append({'id': counter, 'vector': article.vector, 'articles': [article]})
                counter += 1
        return clusters

    def _merge_clusters(self, cluster_a, cluster_b, cluster_pair_list=None, clusters=None):

        cluster_a['articles'].extend(cluster_b['articles'])
        cluster_a['vector'] = self._cluster_vector(cluster_a)

        if clusters is not None:
            for cluster_counter in range(len(clusters)):
                if cluster_b is clusters[cluster_counter]:
                    clusters.pop(cluster_counter)
                    break

        if cluster_pair_list is not None:
            for pair_counter in range(len(cluster_pair_list)):
                pair = cluster_pair_list[pair_counter]
                if cluster_b['id'] is pair['key']['id']:
                    cluster_pair_list.pop(pair_counter)
                    break

            for pair in cluster_pair_list:
                if cluster_b['id'] is pair['target']['id']:
                    newer_pair = self._find_closest_pair(clusters, pair['key'])
                    pair['target'] = newer_pair['target']
                    pair['similarity'] = newer_pair['similarity']

            cluster_pair_list.sort(key=lambda cluster_pair: cluster_pair['similarity'], reverse=True)

    def _build_cluster_pair_list(self, clusters):
        similarity_table = []
        for i in range(len(clusters)):
            pair = self._find_closest_pair(clusters, clusters[i])
            similarity_table.append(pair)
        return sorted(similarity_table, key=lambda cluster_pair: cluster_pair['similarity'], reverse=True)

    def _find_closest_pair(self, clusters, cluster):
        pair = {'key': (cluster, None), 'similarity': 0}
        for j in range(len(clusters)):
            if cluster['id'] == clusters[j]['id']:
                continue
            similarity = self._similarity(cluster, clusters[j])
            if similarity > pair['similarity']:
                pair['similarity'] = similarity
                pair['key'] = cluster
                pair['target'] = clusters[j]
        return pair

    @staticmethod
    def _similarity(cluster_a, cluster_b):
        return dot(cluster_a['vector'], cluster_b['vector'])

    @staticmethod
    def _cluster_vector(cluster):
        return sum([a.vector for a in cluster['articles']]) / len(cluster['articles'])
