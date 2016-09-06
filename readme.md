# Environment
```
Python3
PyCharm
```
若IDE不用PyCharm請記得將專案根目錄加到PYTHONPATH

# Dependency Management

```
pip install -r requirement.txt
```

windows可能會遇到一點麻煩，大部分的麻煩可以[conda](http://conda.pydata.org/docs/intro.html)取代pip來解決


# Word2Vec Model
### 下載預先訓練好的Model
[Wirl NAS](140.124.183.12)上有預先訓練好的Model，目前專案預設使用的是“ngram_300_5_90w"   
NAS登入後請到謝宗廷/Word2Vec Models資料夾下載"ngram_300_5_90w.bin", "ngram_300_5_90w.bin.syn0.npy", "ngram_300_5_90w.bin.syn1neg.npy"這三個檔案，並且放到專案的python_code/model/bin資料夾底下  
一個完整的model會有一個bin檔，如果檔案過大，底層的儲存方式會把他分成好幾個檔案  
model的命名方式為[algorithm名稱_向量長度_mincount_訓練文章數量]  目前資料夾好幾種model可以選擇，學弟們可以比較看看哪一種參數設定效果會比較好

### 訓練自己的Model
開啟python_code/model/train_word2vector.py修改訓練的參數([文件](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec)) and Run it!


# Hot Topic Detection
```
python python_code/main.py yyyy/mm/dd
```

# System Architecture
![系統架構](/document/圖片/System Architecture.png)

btw 系統架構圖利用[draw.io](https://www.draw.io/)畫的，檔案存在[document/System Architecture.xml](/document/System Architecture.xml)  

目前提供好幾種不同的Feature Extraction方法以及不同的Clustering方法  
python_code/agglomerative_clustering_test.py有比較不同的組合的效果，數據整理在log/clustering_log/compare.xml

### FeatureExtractor
FeatureExtractor是一個標準介面(python_code/feature_extractor.py)，提供 fit(articles) function  
articles參數是一個list of article object(這個object該有的欄位定義在model/ptt_article_fetcher), 此function能夠擷取參數article的特徵擷取出來並轉成向量存在article裡面  

FeatureExtractor提供四個Concrete Class實現方法：分別為TFIDF, Title, ContentExtraction, ContentRatioExtraction

* TFIDF(use_idf=True, only_title=False)  
使用傳統的tf-idf vector space取得文章向量  
  
	+ use_idf　　　　　如果為false，就只會用tf term document space model來建文章向量  
	+ only_title　　　　如果為true，就只會用文章的標題來建term matrix  

* Title(model)
只用文章標題斷詞後的詞向量合成文章向量  
  
	+ model　　　　　　word2vec model  

* ContentExtraction(model, method, k, with_weight)  
抓整個文章的關鍵字，利用關鍵字的詞向量來組成文章向量  
  
	+ model　　　　　　word2vec model  
	+ method　　　　　關鍵字擷取的方法, 0為lda, 1為tf-idf based, 2為text rank  
	+ k　　　　　　　　關鍵字擷取的個數  
	+ with_weight　　　需不需要考量關鍵字的權重來合成文章向量  

* ContentRatioExtraction(model, method, k, with_weight, t_ratio, c_ratio)  
除了利用內文關鍵字，也特別加入文章標題來合成文章向量
  
	+ model　　　　　　word2vec model  
	+ method　　　　　關鍵字擷取的方法, 0為lda, 1為tf-idf based, 2為text rank  
	+ k　　　　　　　　關鍵字擷取的個數  
	+ with_weight　　　需不需要考量關鍵字的權重來合成文章向量  
	+ t_ratio　　　　　標題向量的比例 0~1  
	+ c_ratio　　　　　內容向量的比例 0~1, 與t_ratio相加要等於1  

Usage Example
```
FeatureExtractor.TFIDF(use_idf=True, only_title=False).fit(articles)
```
or
```
FeatureExtractor.ContentExtraction(model=model, method=1, k=15, with_weight=True).fit(articles)
```

### Clustering
Clustering使用Hierarchical Agglomerative Clustering(HAC)   
AgglomerativeClustering(python_code/agglomerative_clustering.py)建構子需要提供三個參數:threshold,linkage, similarity，分別指定低於多少相似度不合併、使用的群相似度評估方法、相似度的計算方法。  
目前linkage提供四種方法:LINKAGE_CENTROID, LINKAGE_SINGLE, LINKAGE_AVERAGE, LINKAGE_COMPLETE  
similarity提供兩種方法:SIMILARITY_COSINE, SIMILARITY_DOT  

AgglomerativeClustering提供兩種演算法來分群文章，輸入文章必須先透過FeatureExtractor將特徵轉成向量

+ fit(articles)  
  從文章裡面挑出最相似的兩個文章，若相似度高於門檻值就合併，否則結束分群（分群品質穩定，也是論文採用的方法）
+ quick_fit(articles)
  依序挑一篇文章找出與其最相似的文章合併，若相似度低於門檻則將該文章視為分好的群，重複直到所有的文章都被分好（這是一個貪婪演算法的版本，速度較快但每次分群結果會不一樣，這並不是HAC)

Usage Example
```
clusters = AgglomerativeClustering(threshold=0.75, linkage=LINKAGE_CENTROID, similarity=SIMILARITY_DOT).fit(articles)
```

### Evaluation
首先必須準備人工標記資料，目前提供了三天的人工標記資料（python_code/test/source)，必須先將source轉成json(檔案太大就不上git了)

```
python python_code/test/make_test_data.py
```
很抱歉沒有寫好cls的arg，你需要修改make_test_data.py，並利用make_test_data_from_label_data輸入對應的source來轉json

python_code/clustering_validation提供一個validate_clustering function來評估分群效果  
目前只支援ARI, ARI, Homogeneity, Completeness, V-measure等五個external指標, internal指標的部分還沒有實作完畢（事實上實作上遇到很多問題

```
result = validate_clustering(labeled_clusters, clusters)
```

# Note
這個專案研究方向一波三折，因此保留了一些歷史痕跡留作紀念  
起初只想要用lda作為主題偵測的方法，但寫了一些測試發現不大可行（python_code/test/test_lda)。  
但意外發現lda拿來做keyword expansion效果相當不錯(python_code/model/lda term_expansion)  
  
  
後來開始玩word2vec，寫了一些小測試（python_/test/test_word2vector)覺得可以繼續研究下去。一開始單純只用標題下去分群，效果意外不錯都能夠到80,90以上，後來發現原來是我測試資料來源的問題。為了不手動標記，我用了比較偷懶的方法。先人工找出若干主題以及對應的關鍵字，再用這些關鍵字去搜尋文章標題來標為同一群。但這些標記資料本來就是以標題找來的，所以只用標題來分群效果當然會很好...(舊的code還留在python_code/test/make_test_data.py的make_test_data_from_topic_list)  
  
  
然後那時候為了提高分群效果，試過了許多方式，如分兩次群，或是改變不同的向量合成方式很雜很亂，所以後來把程式整個改寫（舊的留在clustering_v1）  
v1分群用的是貪婪式的版本，每次分群效果都不一樣，所以log才會有平均跟標準差的分析  
那時候覺得HAC實在太慢了，後來改寫了一個版本把算過相似度都暫存起來...合併之後在更新相依的相似度，效率大幅up(覺得我有點天才哈哈哈)。雖然效率跟貪婪版本還是有差，但已經在可以接受的範圍了（因為這個才是hac，就不用再花時間去解釋我一開始寫的版本，當然...我也沒有在論文解釋實際的分群演算法有用到cache加快運算)，而且因為分群結果跑n次都會一樣，分析實驗數據方便很多（這時候就顯得log的平均,極值,標準差分析有點冗贅ＸＤ  
  
v2主要的更動就是加入了linkage跟similarity的觀念還有正港的hac，然後把v1的二次分群都拿掉, 而v3（就是最新的這版）因為覺得v2架構寫得不好看就再refactor一次
