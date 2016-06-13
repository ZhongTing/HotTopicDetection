import time

import gensim
from numpy import array
from sklearn.feature_extraction.text import TfidfVectorizer

from python_code.model.keywords_extraction import keywords_extraction
from python_code.model.my_tokenize.tokenizer import cut


class FeatureExtractor:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.idf_vectorizer = TfidfVectorizer(use_idf=True, tokenizer=cut)

    @staticmethod
    def load_model(model_path):
        t = time.time()
        model = gensim.models.Word2Vec.load(model_path)
        t = int(time.time() - t)
        print('spend {}s to load word2vec model from {}'.format(t, model_path))
        return model

    def fit(self, articles):
        for article in articles:
            article.vector = self._compute_vector(article.title + " " + article.content)
        self.remove_invalid_articles(articles)

    def fit_with_extraction(self, articles, method, topic=10):
        for article in articles:
            keyword_list = keywords_extraction(article, method, topic)
            article.vector = self._compute_vector(keyword_list)
        return self.remove_invalid_articles(articles)

    def fit_with_extraction_ratio(self, articles, t=0.5, c=0.5):
        for article in articles:
            if c == 0:
                article.vector = self._compute_vector(article.title)
            elif t == 0:
                article.vector = self._compute_vector(keywords_extraction(article, 0))
            else:
                title_vector = self._compute_vector(article.title)
                keyword_vector = self._compute_vector(keywords_extraction(article, 0))
                article.vector = title_vector * t + keyword_vector * c
        self.remove_invalid_articles(articles)

    @staticmethod
    def remove_invalid_articles(articles):
        to_be_removed_array = []
        for article in articles:
            if article.vector is None:
                to_be_removed_array.append(article)
        for remove_target in to_be_removed_array:
            articles.remove(remove_target)
        return [a.id for a in to_be_removed_array]

    def _compute_vector(self, input_data, use_idf=False, need_log=False):
        if isinstance(input_data, list):
            tokens = input_data
        else:
            tokens = cut(input_data, using_stopwords=True, simplified_convert=True, log=need_log)

        if len(tokens) > 0 and (tokens[-1] in ['八卦', '卦']):
            del tokens[-1]
        v1 = [self.model[word] for word in tokens if word in self.model]
        if len(v1) is 0:
            print('invalid article:', input_data)
            return None
        return array(v1, float).mean(axis=0)
