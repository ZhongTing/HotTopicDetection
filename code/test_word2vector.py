import gensim.models

model_path = 'bin/word2vecmodel_10000.bin'
model = gensim.models.Word2Vec.load(model_path)

print(model)
print(model.similarity('死刑', 'FB'))
print(model.doesnt_match(["大學", "小學", "八卦"]))
