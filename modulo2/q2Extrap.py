# CÓDIGO USANDO EXTRAPOLAÇÃO, FUNÇÃO DE ATIVAÇÃO TANH E FUNÇÃO DE OTIMIZAÇÃO 
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

x = np.linspace(0, 10, 100)       # gera 100 valores no intervalo [0-10]
y = f2(x)                         # joga na função e computa o valor de f2 nestes 100 valores

print(len(x), "\nValores de x =\n", x)
print(len(y), "\nValores de y =\n", y)

# Definindo o modelo com a função de ativação 'tanh' e otimizador SGD
model = tf.keras.Sequential(
    [
        keras.Input(shape=(1,)),                            # Define a entrada do modelo com uma forma de vetor de 1 dimensão
        keras.layers.Dense(units=64, activation='tanh'),    # Camada densa com função de ativação 'tanh'
        keras.layers.Dense(units=1)                         # Camada de saída com 1 neurônio, sem função de ativação
    ]
)

model.compile(
    optimizer="sgd",                                # Otimizador SGD
    loss="mean_squared_error"                       # Função de perda comumente usada em problemas de regressão
)

model.fit(x, y, epochs=400, verbose=0)              # Treinando o modelo com os dados de x e y e 400 épocas

model.summary()

# Prediction do Modelo e VALOR REAL dentro do intervalo de treino
prediction = model.predict([17])
real_value = f2(17)
print(f"Optimizer: SGD\nPrediction (in interval): {prediction[0][0]}\nReal Value (in interval): {real_value}")

# Avaliação do modelo dentro do intervalo de treino
x_val = np.linspace(0, 10, 63)
y_val = f2(x_val)

loss = model.evaluate(x=x_val, y=y_val, verbose=0)
print(f"Loss (in interval): {loss}\n")

# Avaliação do modelo fora do intervalo de treino
x_extrap = np.linspace(10, 20, 50)  # Valores de 10 a 20
y_extrap = f2(x_extrap)

extrap_loss = model.evaluate(x=x_extrap, y=y_extrap, verbose=0)
print(f"Loss (extrapolation): {extrap_loss}")

# Prediction do Modelo e VALOR REAL fora do intervalo de treino
prediction_extrap = model.predict([25])
real_value_extrap = f2(25)
print(f"Prediction (extrapolation): {prediction_extrap[0][0]}\nReal Value (extrapolation): {real_value_extrap}")
