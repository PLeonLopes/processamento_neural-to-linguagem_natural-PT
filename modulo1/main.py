import spacy
import pt_core_news_sm
spacyPT = pt_core_news_sm.load()

spacy.__version__

entrada = spacyPT("Mais vale um asno que me carregue que um cavalo que me derrube.")

print(f"ENTRADA CRUA: {entrada}\n")

print(f"ENTRADA EM TOKEN: {entrada.text.split()}\n")

print(f"ENTRADA EM TOKEN SEM VÍRGULAS OU PONTOS: {[token for token in entrada]}\n")       # lista de OBJETOS da classe TOKEN]

print(f"ENTRADA EM TOKEN SEM VÍRGULAS OU PONTOS: {[token.text for token in entrada]}\n")  # lista de STRINGS (pegando o texto do objeto)

print(f"ENTRADA EM TOKEN SEM PONTUAÇÃO: {[token.text for token in entrada if not token.is_punct]}\n")   # lista de STRINGS sem pontuação

print(f"ETIQUETAGEM MORFOSSINTÁTICA: {[(token.text, token.pos_) for token in entrada]}\n")      # mostra TOKENS e morfosintática deles

print(f"LEMA DOS VERBOS: {[token.lemma_ for token in entrada if token.pos_ == 'VERB']}\n")      # mostra só os tokens dos verbos (infinitivo)

# RECONHECIMENTO DE ENTIDADADES NOMEADAS

texto2 = spacyPT("A CBF fez um pedido de análise ao Comitê de Apelações da FIFA a fim de diminuir a pena do atacante Neymar, suspenso da Copa América pela Conmebol.")
print("RECONHECIMENTO DE ENTIDADES NOMEADAS\n")
print(f"TEXTO COMPLETO: {texto2.ents}\n")
print(f"RECONHECIMENTO DAS ENTIDADES: {[(entidade,entidade.label_) for entidade in texto2.ents]}\n")


