from gensim.models import KeyedVectors

# Carregar o modelo word2vec
model = KeyedVectors.load_word2vec_format("modulo4/data/word2vec_200k.txt", binary=False)

if model:
    print("word2vec carregado com sucesso!")
else:
    print("erro ao carregar word2vec")


vies = model.most_similar("idoso")
print(vies)