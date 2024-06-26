import spacy
from spacy import displacy
import pt_core_news_sm
spacyPT = pt_core_news_sm.load()

spacy.__version__

entrada = spacyPT("Mas, já que falei nos meus dois tios, deixem-me fazer aqui um curto esboço genealógico.        O fundador de minha família foi um certo Damião Cubas, que floresceu na primeira metade do século XVIII. Era tanoeiro de ofício, natural do Rio de Janeiro, onde teria morrido na penúria e na obscuridade, se somente exercesse a tanoaria. Mas não; fez-se lavrador, plantou, colheu, permutou o seu produto por boas e honradas patacas, até que morreu, deixando grosso cabedal a um filho, o licenciado Luís Cubas. Neste rapaz é que verdadeiramente começa a série de meus avós -- dos avós que a minha família sempre confessou -  porque o Damião Cubas era afinal de contas um tanoeiro, e talvez mau tanoeiro, ao passo que o Luís Cubas estudou em Coimbra, primou no Estado, e foi um dos amigos particulares do vice-rei conde da Cunha.        Como este apelido de Cubas lhe cheirasse excessivamente a tanoaria, alegava meu pai, bisneto do Damião, que o dito apelido fora dado a um cavaleiro, herói nas jornadas da Africa, em prêmio da façanha que praticou arrebatando trezentas cubas ao mouros. Meu pai era homem de imaginação; escapou à tanoaria nas asas de um calembour. Era um bom caráter, meu pai, varão digno e leal como poucos. Tinha, é verdade, uns fumos de pacholice; mas quem não é um pouco pachola nesse mundo? Releva notar que ele não recorreu à inventiva senão depois de experimentar a falsificação; primeiramente, entroncou-se na família daquele meu famoso homônimo, o capitão-mor Brás Cubas, que fundou a vila de São Vicente, onde morreu em 1592, e por esse motivo é que me deu o nome de Brás. Opôs-se-lhe, porém, a família do capitão-mor, e foi então que ele imaginou as trezentas cubas mouriscas.        Vivem ainda alguns membros de minha família, minha sobrinha Venância, por exemplo, o lírio-do-vale, que é a flor das damas do seu tempo; vive o pai, o Cotrim, um sujeito que... Mas não antecipemos os sucessos; acabemos de uma vez com o nosso emplasto. ")

print(f"TEXTO COMPLETO: \n {entrada} \n")

frases = [frase for frase in entrada.sents]
print(f"FRASE ESCOLHIDA: {frases[2]}")

# Visualizar o mapa de dependências da frase escolhida, abre no navegador (0.0.0.0:5000)
# obs: PODE OCORRER O ERRO DA PORTA JÁ ESTAR SENDO USADA, CASO NÃO CONSIGA ABRIR, UTILIZE: "localhost:5000" -> (abrir navegador e digitar localhost:num_porta)
displacy.serve(frases[2], style="dep")
