from gensim.models import KeyedVectors

# Carregar o modelo word2vec
model = KeyedVectors.load_word2vec_format("modulo4/data/word2vec_200k.txt", binary=False)

if model:
    print("word2vec carregado com sucesso!")
else:
    print("erro ao carregar word2vec")

'''
ANALOGIAS PARA TESTE:
homem : mulher :: rainha : x
rei : homem :: mulher : x
paris : frança :: berlim : x
alemão : alemanha :: inglês : x
Brasil : Alemanha :: Berlim : x
'''
# Lista com exemplos de analogias
analogias = [
    (["homem", "rainha"], ["mulher"]),          # homem : mulher :: rainha : x
    (["rei", "mulher"], ["homem"]),             # rei : homem :: mulher : x
    (["paris", "alemanha"], ["frança"]),        # paris : frança :: berlim : x
    (["alemão", "inglaterra"], ["alemanha"])    # alemão : alemanha :: inglês : x
]

# Função para encontrar a palavra análoga
def encontrar_analogia(positivas, negativas):
    try:
        similar = model.most_similar(positive=positivas, negative=negativas, topn=1)
        return similar[0][0]
    except KeyError as e:
        print(f"Palavra não encontrada no vocabulário: {e}")
        return None

# Testar cada analogia
for pos, neg in analogias:
    resultado = encontrar_analogia(pos, neg)
    if resultado:
        print(f"{pos[0]} : {neg[0]} :: {pos[1]} : {resultado}")

# mais analogias para teste
mais_analogias = [
    (["brasil", "berlim"], ["alemanha"]),       # Brasil : Alemanha :: Berlim : x
]

for pos, neg in mais_analogias:
    resultado = encontrar_analogia(pos, neg)
    if resultado:
        print(f"{pos[0]} : {neg[0]} :: {pos[1]} : {resultado}")
