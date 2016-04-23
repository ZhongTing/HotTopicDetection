import ptt_article_fetcher
import tokenizer

articles = ptt_article_fetcher.fetch_articles('', number=10, page=6)
using_stopword = False

equals_tokens = []
for article in articles:
    token1 = tokenizer.cut(article.title, using_stopword, True)
    token2 = tokenizer.cut(article.title, using_stopword, False)
    if token1 == token2:
        equals_tokens.append(token1)
    else:
        print('經轉換' + str(token1))
        print('未轉換' + str(token2))

for i in equals_tokens:
    print(i)
