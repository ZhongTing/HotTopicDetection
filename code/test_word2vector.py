import gensim.models
from ptt_article_fetcher import fetch_articles
from tokenizer import cut

model_path = 'bin/word2vecmodel_780000.bin'
#model_path = 'bin/word2vecmodel_788051.bin'
model = gensim.models.Word2Vec.load(model_path)


def similarity_test(arg1='台灣', arg2='中國'):
	print(model.similarity(arg1, arg2))

def doesnt_match_test():
	data_set = [
		["早餐", "午餐", "晚餐", "宵夜", "車禍"],
		["國文", "英文", "數學", "物理", "電腦"],
		["爸爸", "媽媽", "書包"]
	]
	for data in data_set:
		print(data)
		print(model.doesnt_match(data))


def most_similar_test(data_set=None):
	if data_set==None:
		data_set = [
			{'positive':['台灣', '中文'], 'negative':['美國']},
			{'positive':['桌子', '椅子']},
			{'positive':['奶茶', '雞排']},
			{'positive':['爸爸', '男'], 'negative':['女']},
			{'positive':['弟弟', '男'], 'negative':['女']},
			{'positive':['爺爺', '男'], 'negative':['奶奶']},
		]

	for data in data_set:
		positive = data.get('positive',[])
		negative = data.get('negative',[])
		print('正:'+"/".join(positive) + '     反:'+"/".join(negative))
		print(model.most_similar(positive, negative, topn=5))


def n_similarity_test():
	sentence_list = ['馬總統走光照 蔡正元:經專家鑑定為光影',
		'馬走光照瘋傳總統府譴責',
		'萬安演習著恐龍裝逛大街 男遭裁罰',
		'舉債也最低... 「六都還款王」第2名令人',
		'【北捷殺人案】鄭捷判死定讞5大理由曝光',
		'地震',
		'日本紅十字會：捐款不用手續費　善款100%',
		'日本熊本強震 屏縣府擬捐香蕉賑災',
		'有沒有日本重新定義島的八卦',
		'日本只在利益不衝突時才是朋友'
	]

	tokens_list = [cut(sentence) for sentence in sentence_list]

	for i in range(1, len(tokens_list)):
		print(tokens_list[i-1])
		print(tokens_list[i])
		print(model.n_similarity(tokens_list[i], tokens_list[i-1]))


#most_similar_test()
#n_similarity_test()
#similarity_test('日文', '日語')
#similarity_test('伊斯蘭教', '佛教')
doesnt_match_test()