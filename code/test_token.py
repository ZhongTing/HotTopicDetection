from code import ptt_article_fetcher
from code.tokenizer import cut

articles = ptt_article_fetcher.fetch_articles('', number=10, page=6)
using_stopwords = False

equals_tokens = []
for article in articles:
    token1 = cut(article.title, using_stopwords, True)
    token2 = cut(article.title, using_stopwords, False)
    if token1 == token2:
        equals_tokens.append(token1)
    else:
        print('經轉換' + str(token1))
        print('未轉換' + str(token2))

for i in equals_tokens:
    print(i)
