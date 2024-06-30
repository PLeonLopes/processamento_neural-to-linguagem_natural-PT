import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split

tf.__version__

# PERCEPTRON -> Uma "rede neural" de um só neurônio
'''
No exemplo: Teremos uma rede mais simples possível, com uma só entrada e uma só saída, sem ativação.
Temos 100 dados que serão usados para treinar 300 épocas do percéptron (época é basicamente um ciclo, ou seja, se tiver 300 épocas, o treinamento ocorrerá por 300 ciclos).
Vamos utilizar o modelo percéptron para aprender uma simples regressão linear, o objetivo é fazê-lo aprender uma simples equação linear e tambem se acostumar com a sintaxe e funcionamento do TensorFlow
'''

def f1(x):
    # Função a ser aprendida
    return 5 + 10*x

xs = np.linspace(0,10,100)      # gera 100 valores no intervalo [0-10]
ys = f1(xs)                     # joga na função e computa o valor de f1 nestes 100 valores
print(len(xs), "\nxs=", xs)
print(len(ys), "\nys=", ys)

# definindo, compilando e treinando o modelo
model = tf.keras.Sequential(
    [   
        keras.Input(shape=(1,)),        # Define a entrada do modelo com uma forma de vetor de 1 dimensão
        keras.layers.Dense(units=1),    # Adiciona uma camada densa (totalmente conectada) com 1 neurônio E SEM FUNÇÃO DE ATIVAÇÃO
    ]
)

model.compile(
    optimizer="adam",                # Mudança de SDG para ADAM
    loss="mean_squared_error")       # função de perda comumente usada em problemas de regressão

model.fit(xs,ys,epochs=300)     # treinando o modelo com os dados de xs e ys e 300 épocas

model.summary()     # imprime o resumo do modelo

print(f"Prediction: {str(model.predict([17]))}\nReal Value: {str(f1(17))}")     # Prediction do Modelo e VALOR REAL

# Retorna o loss (custo) da avaliação, definido na compilação. Nesse caso, o valor reportado é o erro quadrático médio (MSE). 
# É util para avaliar a performance do modelo
val = np.linspace(0,10,63)
model.evaluate(x=val, y=f1(val))


# FUNÇÃO NÃO LINEAR (exemplo)
def f2(x):
    # Funcao não linear a ser aprendida
    return (x**2 + x*3 + 4)/200

x = np.linspace(0,10,100)
y = f2(x)
