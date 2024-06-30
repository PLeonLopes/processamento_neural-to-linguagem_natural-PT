# Testes com a Função de Otimização RMSprop (Root Mean Square Propagation):
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

# Definindo o modelo com a função de ativação 'tanh' e otimizador RMSprop
model = tf.keras.Sequential(
    [
        keras.Input(shape=(1,)),                    # Define a entrada do modelo com uma forma de vetor de 1 dimensão
        keras.layers.Dense(units=64, activation='tanh'),  # Camada densa com função de ativação 'tanh'
        keras.layers.Dense(units=1)                 # Camada de saída com 1 neurônio, sem função de ativação
    ]
)

model.compile(
    optimizer="rmsprop",                            # Otimizador RMSprop
    loss="mean_squared_error"                       # Função de perda comumente usada em problemas de regressão
)

model.fit(x, y, epochs=400, verbose=0)              # Treinando o modelo com os dados de x e y e 400 épocas

model.summary()

# Prediction do Modelo e VALOR REAL
prediction = model.predict([17])
real_value = f2(17)
print(f"Optimizer: RMSprop\nPrediction: {prediction[0][0]}\nReal Value: {real_value}") 

# Avaliação do modelo
x_val = np.linspace(0, 10, 63)
y_val = f2(x_val)

loss = model.evaluate(x=x_val, y=y_val, verbose=0)
print(f"Loss: {loss}\n")
