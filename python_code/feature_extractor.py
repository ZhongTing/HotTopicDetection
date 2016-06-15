import time

import gensim
from numpy import array
from sklearn.feature_extraction.text import TfidfVectorizer

from python_code.model.keywords_extraction import keywords_extraction
from python_code.model.my_tokenize.tokenizer import cut


class FeatureExtractor:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    @staticmethod
    def load_model(model_path):
        t = time.time()
        model = gensim.models.Word2Vec.load(model_path)
        t = int(time.time() - t)
        print('spend {}s to load word2vec model from {}'.format(t, model_path))
        return model

    @staticmethod
    def idf_vectorizer(articles, use_idf):
        if use_idf:
            try:
                documents = [article.title + " " + article.content for article in articles]
            except TypeError:
                documents = articles
            tfidf_vectorizer = TfidfVectorizer(use_idf=True, tokenizer=cut)
            tfidf_vectorizer.fit(documents)
            return tfidf_vectorizer
        else:
            return None

    def fit(self, articles, use_idf=False):
        tfidf_vectorizer = self.idf_vectorizer(articles, use_idf)
        for article in articles:
            article.vector = self._compute_vector(article.title + " " + article.content, tfidf_vectorizer)
        self.remove_invalid_articles(articles)

    def fit_with_extraction(self, articles, method, topic=10, use_idf=False, with_weight=False):
        documents = []
        keyword_list_table = {}
        for article in articles:
            keyword_list = keywords_extraction(article, method, topic, with_weight=with_weight)
            keyword_list_table[article.id] = keyword_list
            if with_weight:
                documents.append(' '.join([keyword[0] for keyword in keyword_list]))
            else:
                documents.append(' '.join(keyword_list))
        tfidf_vectorizer = self.idf_vectorizer(documents, use_idf)
        for article in articles:
            keyword_list = keyword_list_table[article.id]
            article.vector = self._compute_vector(keyword_list, tfidf_vectorizer)
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

    def _compute_vector(self, input_data, tfidf_vectorizer=None):
        weights = None
        if isinstance(input_data, list):
            if isinstance(input_data[0], tuple):
                tokens = [data_tuple[0] for data_tuple in input_data]
                weights = [data_tuple[1] for data_tuple in input_data]
            else:
                tokens = input_data
        else:
            tokens = cut(input_data, using_stopwords=True, simplified_convert=True)

        if len(tokens) > 0 and (tokens[-1] in ['八卦', '卦']):
            del tokens[-1]
        v1 = []
        if tfidf_vectorizer is not None:
            idf_table = self.build_idf_table(tfidf_vectorizer)
        for word in tokens:
            if word in self.model:
                word_vector = self.model[word]
                if weights:
                    weight = weights[tokens.index(word)]
                    word_vector = word_vector * weight
                if tfidf_vectorizer is not None and word in idf_table:
                    word_vector = word_vector * idf_table[word]
                v1.append(word_vector)
        if len(v1) is 0:
            print('invalid article:', input_data)
            return None

        # v1 = [self.model[word] for word in tokens if word in self.model]
        if tfidf_vectorizer is None:
            return array(v1, float).mean(axis=0)
        else:
            return sum(v1)

    @staticmethod
    def build_idf_table(tfidf_vectorizer):
        table = {}
        term_list = tfidf_vectorizer.get_feature_names()
        for i in range(len(term_list)):
            table[term_list[i]] = tfidf_vectorizer.idf_[i]
        return table
