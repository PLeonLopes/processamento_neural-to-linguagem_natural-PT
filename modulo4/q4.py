from gensim.models import KeyedVectors

# Carregar o modelo word2vec
model = KeyedVectors.load_word2vec_format("modulo4/data/word2vec_200k.txt", binary=False)

if model:
    print("word2vec carregado com sucesso!")
else:
    print("erro ao carregar word2vec")

# Lista de analogias para testar, incluindo o resultado esperado
analogias = [
    (["homem", "rainha"], ["mulher"], "rei"),      # homem : mulher :: rainha : rei
    (["rei", "mulher"], ["homem"], "rainha"),      # rei : homem :: mulher : rainha
    (["paris", "alemanha"], ["frança"], "berlim"), # paris : frança :: alemanha : berlim
    (["alemão", "inglaterra"], ["alemanha"], "inglês") # alemão : alemanha :: inglaterra : inglês
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
for pos, neg, esperado in analogias:
    resultado = encontrar_analogia(pos, neg)
    if resultado and resultado != esperado:
        print(f"{pos[0]} : {neg[0]} :: {pos[1]} : {esperado} (Esperado)")
        print(f"Valor errado encontrado: {resultado}\n")

# Adicionar suas próprias analogias para testar
# exemplo: analogia customizada com resultado esperado
custom_analogias = [
    (["brasil", "berlim"], ["alemanha"], "brasilia"), # Brasil : Alemanha :: Berlim : Brasilia
]

for pos, neg, esperado in custom_analogias:
    resultado = encontrar_analogia(pos, neg)
    if resultado and resultado != esperado:
        print(f"{pos[0]} : {neg[0]} :: {pos[1]} : {esperado} (Esperado)")
        print(f"Valor errado encontrado: {resultado}\n")
