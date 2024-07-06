from gensim.models import KeyedVectors

# Carregar o modelo word2vec
model = KeyedVectors.load_word2vec_format("modulo4/data/word2vec_200k.txt", binary=False)

if model:
    print("word2vec carregado com sucesso!")
else:
    print("erro ao carregar word2vec")

# Função para encontrar palavras similares
def encontrar_similares(palavra):
    try:
        similares = model.most_similar(positive=[palavra])      # pega a palavra escolhida
        print(f"Palavras similares a '{palavra}': {[sim[0] for sim in similares]}")
        return [sim[0] for sim in similares]
    except KeyError:
        print(f"A palavra '{palavra}' não está no vocabulário.")        # checa por erros
        return []

# Função para calcular distância entre palavras
def calcular_distancia(palavra1, palavra2):
    try:
        distancia = model.distance(palavra1, palavra2)      # pega as duas palavras
        print(f"Distância entre '{palavra1}' e '{palavra2}': {distancia}")
        return distancia
    except KeyError:
        print(f"Uma das palavras '{palavra1}' ou '{palavra2}' não está no vocabulário.")        # checa por erros
        return float('inf')

palavras_teste = ["manga", "maça", "feliz"]     # palavras escolhidas

resultado = []

# Encontrar sinônimos e antônimos para cada palavra de teste
for palavra in palavras_teste:
    print(f"\nAnalisando a palavra: {palavra}")
    similares = encontrar_similares(palavra)
    if similares:
        sinonimo = similares[0]
        for ant in similares[1:]:
            dist_sin = calcular_distancia(palavra, sinonimo)
            dist_ant = calcular_distancia(palavra, ant)
            if dist_ant < dist_sin:
                resultado.append((palavra, sinonimo, ant, dist_sin, dist_ant))

# RESULTADOS
for res in resultado:
    print(f"\nPalavra: {res[0]}")
    print(f"Sinônimo: {res[1]} (Distância: {res[3]})")
    print(f"Antônimo: {res[2]} (Distância: {res[4]})")

if not resultado:
    print("Nenhum resultado encontrado que satisfaça a condição.")      # caso não seja encontrado nenhum resultado final
