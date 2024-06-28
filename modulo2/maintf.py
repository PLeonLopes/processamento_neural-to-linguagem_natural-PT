import tensorflow as tf
import numpy as np

# KERAS
from tensorflow import keras
from keras import layers

tf.__version__

# TENSORES -> (array retangular n-dimensional (uma matriz com n índices) e o objeto mais basico do TensorFlow. É usado para representar seus dados e fazê-los fluir de operação em operação, dai que vem o nome -> TensorFlow

print(f"\nVetor de 2 matrizes 3x4 e vazio: \n{tf.zeros([2,3,4])}")     # um vetor com 2 matrizes 3 x 4, zerado

print(f"\nVetor de 2 matrizes 3x4 e preenchido com 1's: \n{tf.ones([2,3,4])}")      # um vetor com 2 matrizes, 3 x 4, preenchido com 1's

print(f"\nVetor de INT de tamanho 3: \n{tf.constant([1,2,3])}")      # esse tensor pode ser visto como um vetor de inteiros de tamanho 3, constante

print(f"\nVetor de LONG de tamanho 3: \n{tf.constant(np.array([1,2,3]))}")      # mesma coisa de cima, mas o vetor é de tipo long (usando numpy)

print(f"\nVetor de UNSIGNED LONG de tamanho 3: \n{tf.constant([1,2,3], dtype=np.uint)}")     # mesa coisa de cima, mas o vetor é de tipo unsigned long (usando numpy)

'''
TENSORES CONSTANTES -> estão "escritos em pedra", seus valores não podem ser mais alterados mas podem ser usados como inputs para funções.

TENSORES VARIÁVEIS -> podem ter seus valores alterados ao realizarmos operações sobre eles, o modulo tf.keras os utiliza internamente para representar os pesos de um modelo. Ao inicializarmos um tensor variavel seu dtype e shape são fixados. Para mais informações, checar este guia (https://www.tensorflow.org/guide/variable?hl=pt-br)

RAGGED TENSORS -> Basicamente tensores não retangulares, onde nem todas as dimensões dos elementos são iguais (checar ipynb para fotos)

SPARCE TENSORS -> Tensores onde a maioria dos seus elementos são nulos        (checar ipynb para fotos)
'''

# Exemplos de vetores constantes:
a = tf.constant([1,2,3,4,5])
b = tf.constant([5,4,3,2,1])
c = a + b
print(f"\nExemplo de soma de vetores constantes: \n{c}")      # soma de vetores constantes

print(f"\nConversão de tensor para array (numpy): \n{c.numpy()}")        # converte o tensor para um array de numpy

print(f"\nSlicing mostrando posições 1 a 4: \n{a[1:4].numpy()}")      # slicing em tensors funciona da mesma maneira que listas.py -> ex: código mostra posições 1 a 4 do array

# Exemplos de vetores variáveis:
f = tf.Variable([1,2,3,4,5])
f[1].assign(19)     # tensores do tipo variaveis podem ter seus conteudos alterados
print(f"\nTensor variável com valor adicionado: \n{f}")
'''
    Tentativa de adição de um valor em um tesor constantes: a[1].assign(17)
    OCORRERÁ UM ERRO, pois tensores constantes são NÃO MUTÁVEIS
'''

# KERAS -> o TensorFlow segue a API do keras. Utilizando as camadas já existentes no Keras podemos construir um modelo ao ligarmos elas de maneira sequencial. Uma vez que o modelo esteja definido, basta compilá-lo e então treiná-lo. 
# A seguir temos um exemplo minimal de uma rede neural feedforward. Documentação: (https://keras.io/guides/sequential_model/) e (https://keras.io/guides/functional_api/)

model = keras.Sequential(                           # Cria um modelo de rede neural
    [   keras.Input(shape=(4,)),                    # Define a entrada do modelo com uma forma de vetor de 4 dimensões
        layers.Dense(2, activation="relu"),         # Adiciona uma camada densa (totalmente conectada) com 2 neurônios e função de ativação ReLU
        layers.Dense(3, activation="relu"),         # Adiciona outra camada densa com 3 neurônios e função de ativação ReLU
        layers.Dense(4, activation="sigmoid"),      # Adiciona a camada de saída com 4 neurônios e função de ativação Sigmoid.
    ]
)

model.summary()     # imprime o resumo do modelo

model.compile(                          # optimizer (atualiza os pesos do modelo com base na função de perda calculada)
    optimizer= 'rmsprop',               # (Root Mean Square Propagation), mantêm uma taxa de aprendizado adaptativa, ajustando a taxa de acordo com a média dos quadrados dos gradientes anteriores. Existem vários tipos de otimizadores.
    loss='binary_crossentropy',         # Calcular a perda, essa é usada para problemas de classificação binária (0 ou 1)
    metrics = None)                     # Não utiliza nenhuma métrica a mais

# model.fit(x= dados_treino, y= labels_treino, batch_size=32, epochs=300)     # treinamento, com um conjunto de dados que seriam inseridos em x e y

model.summary()
