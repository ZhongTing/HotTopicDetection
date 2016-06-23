from numpy import array

from python_code.model.my_tokenize.tokenizer import cut
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFeatureExtractor:
    def __init__(self, use_idf):
        self.vectorizer = TfidfVectorizer(use_idf=use_idf, tokenizer=cut)

    def fit(self, articles, use_content=False, title_ratio=1, content_ratio=1):
        if use_content is False:
            documents = [article.title for article in articles]
        else:
            documents = []
            for article in articles:
                t = (article.title + ' ') * title_ratio
                c = (article.content + ' ') * content_ratio
                documents.append(t + c)
        x = self.vectorizer.fit_transform(documents)
        a_counter = 0
        for i in x:
            # noinspection PyUnusedLocal
            v = [0 for k in range(len(self.vectorizer.get_feature_names()))]
            for j in range(len(i.indices)):
                v[i.indices[j]] = i.data[j]
            articles[a_counter].vector = array(v, float)
            a_counter += 1
