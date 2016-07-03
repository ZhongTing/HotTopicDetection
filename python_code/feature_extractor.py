from abc import abstractmethod

import time

import gensim
from numpy import array
from sklearn.feature_extraction.text import TfidfVectorizer

from python_code.model.keywords_extraction import keywords_extraction
from python_code.model.my_tokenize.tokenizer import cut


def load_model(model_path):
    t = time.time()
    model = gensim.models.Word2Vec.load(model_path)
    t = int(time.time() - t)
    print('spend {}s to load word2vec model from {}'.format(t, model_path))
    return model


class FeatureExtractor:
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def fit(self, articles):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def args(self):
        pass

    def _compute_vector(self, input_data):
        weights = None
        if isinstance(input_data, list):
            if len(input_data) == 0:
                tokens = []
            elif isinstance(input_data[0], tuple):
                tokens = [data_tuple[0] for data_tuple in input_data]
                weights = [data_tuple[1] for data_tuple in input_data]
            else:
                tokens = input_data
        else:
            tokens = cut(input_data, using_stopwords=True, simplified_convert=True)

        if len(tokens) > 0 and (tokens[-1] in ['八卦', '卦']):
            del tokens[-1]
        v1 = []
        for word in tokens:
            if word in self._model:
                word_vector = self._model[word]
                if weights:
                    weight = weights[tokens.index(word)]
                    word_vector = word_vector * weight
                v1.append(word_vector)
        if len(v1) is 0:
            print('invalid article:', input_data)
            return None
        return sum(v1)

    @staticmethod
    def remove_invalid_articles(articles):
        to_be_removed_array = []
        for article in articles:
            if article.vector is None:
                to_be_removed_array.append(article)
        for remove_target in to_be_removed_array:
            articles.remove(remove_target)
        return [a.id for a in to_be_removed_array]


class TFIDF(FeatureExtractor):
    def __init__(self, use_idf=True, only_title=False):
        FeatureExtractor.__init__(self, None)
        self.vectorizer = TfidfVectorizer(use_idf=use_idf, tokenizer=cut)
        self.use_idf = use_idf
        self.only_title = only_title

    def name(self):
        return 'TF-IDF'

    def args(self):
        return 'use_idf={} only_title={}'.format(self.use_idf, self.only_title)

    def fit(self, articles):
        documents = []
        for article in articles:
            document = article.title if self.only_title is True else article.title + ' ' + article.content
            documents.append(document)

        x = self.vectorizer.fit_transform(documents)
        a_counter = 0
        for i in x:
            # noinspection PyUnusedLocal
            v = [0 for k in range(len(self.vectorizer.get_feature_names()))]
            for j in range(len(i.indices)):
                v[i.indices[j]] = i.data[j]
            articles[a_counter].vector = array(v, float)
            a_counter += 1


class Title(FeatureExtractor):
    def __init__(self, model):
        FeatureExtractor.__init__(self, model)

    def fit(self, articles):
        for article in articles:
            article.vector = self._compute_vector(article.title)
        self.remove_invalid_articles(articles)
        pass

    def name(self):
        return 'Title'

    def args(self):
        return None


class ContentExtraction(FeatureExtractor):
    def __init__(self, model, method, k, with_weight):
        FeatureExtractor.__init__(self, model)
        self.method = method
        self.k = k
        self.with_weight = with_weight

    def fit(self, articles):
        for article in articles:
            keyword_list = keywords_extraction(article, self.method, self.k, with_weight=self.with_weight)
            article.vector = self._compute_vector(keyword_list)
        self.remove_invalid_articles(articles)

    def name(self):
        return "Content Extraction"

    def args(self):
        return 'method={} k={} with_weight={}'.format(self.method, self.k, self.with_weight)


class ContentRatioExtraction(FeatureExtractor):
    def __init__(self, model, method, k, with_weight, t_ratio, c_ratio):
        FeatureExtractor.__init__(self, model)
        self.method = method
        self.k = k
        self.with_weight = with_weight
        self.t_ratio = t_ratio
        self.c_ratio = c_ratio

    def fit(self, articles):
        for article in articles:
            keyword_list = keywords_extraction(article, self.method, self.k, with_weight=self.with_weight)
            title_vector = self._compute_vector(article.title)
            content_vector = self._compute_vector(keyword_list)
            if title_vector is None:
                article.vector = content_vector if self.c_ratio != 0 else None
            elif content_vector is None:
                article.vector = title_vector if self.t_ratio != 0 else None
            else:
                article.vector = title_vector * self.t_ratio + content_vector * self.c_ratio
        self.remove_invalid_articles(articles)

    def name(self):
        return "Content Extraction Ratio"

    def args(self):
        return 'method={} k={} with_weight={} t={} c={}'.format(self.method, self.k, self.with_weight, self.t_ratio,
                                                                self.c_ratio)
