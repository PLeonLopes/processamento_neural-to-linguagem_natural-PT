from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("modulo4/data/word2vec_200k.txt")      # Carrega word2vec

if (model):
    print("word2vec carregado com sucesso!")
else:
    print("erro ao carregar word2vec")