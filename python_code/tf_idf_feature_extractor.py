from numpy import array

from python_code.model.my_tokenize.tokenizer import cut
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(use_idf=True, tokenizer=cut)

    def fit(self, articles):
        documents = [article.title + " " + article.content for article in articles]
        x = self.vectorizer.fit_transform(documents)
        a_counter = 0
        for i in x:
            # noinspection PyUnusedLocal
            v = [0 for k in range(len(self.vectorizer.get_feature_names()))]
            for j in range(len(i.indices)):
                v[i.indices[j]] = i.data[j]
            articles[a_counter].vector = array(v, float)
            a_counter += 1
