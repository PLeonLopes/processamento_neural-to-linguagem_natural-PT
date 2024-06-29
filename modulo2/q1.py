import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split

tf.__version__

def f2(x):
    # função não-linear a ser aprendida
    return (x**2 + x*3 + 4)/200

x = np.linspace(0,10,100)       # gera 100 valores no intervalo [0-10]
y = f2(x)                       # joga na função e computa o valor de f1 nestes 100 valores

print(len(x), "\nValores de x =\n", x)
print(len(y), "\nValores de y =\n", y)

# definindo, compilando e treinando o modelo
model = tf.keras.Sequential(
    [    
        keras.Input(shape=(1,)),        # Define a entrada do modelo com uma forma de vetor de 1 dimensão
        keras.layers.Dense(units=1),    # Adiciona uma camada densa (totalmente conectada) com 1 neurônio E SEM FUNÇÃO DE ATIVAÇÃO'''
    ]
)

model.compile(
    optimizer="adam",                   # (Adaptive Moment Estimation) 
    loss="mean_squared_error")          # função de perda comumente usada em problemas de regressão

model.fit(x,y,epochs=400)               # treinando o modelo com os dados de x e y e 400 épocas

model.summary()

print(f"Prediction: {str(model.predict([17]))}\nReal Value: {str(f2(17))}")     # Prediction do Modelo e VALOR REAL

# Retorna o loss (custo) da avaliação, definido na compilação. Nesse caso, o valor reportado é o erro quadrático médio (MSE). 
# É util para avaliar a performance do modelo
x_val = np.linspace(0,10,63)
y_val = f2(x_val)

model.evaluate(x=x_val,y=y_val)

'''
    PERGUNTA 1 -> Defina as camadas para esta rede neural e treine seu modelo, note que a saída unitária não deve ter função de ativação (porque é um problema de regressão).

    RESPOSTA 1 -> As camadas foram definidas na linha 25 deste programa ("keras.layers.Dense(units=1),"). A saída unitária não deve ter função de ativação porque este é um problema de regressão
'''