import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.datasets import make_circles
from numpy import where
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomUniform

tf.__version__

# EXPLODING

def f1(x):
    return 5 + 10*x

xs = [x for x in range(100)]
ys = [f1(x) for x in range(100)]

opt = keras.optimizers.SGD()
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer=opt, loss="mean_squared_error")
model.fit(xs,ys,epochs=400)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Seu código aqui
'''
Ocorreu um problema na realizações das questões abaixo, devido a versão do Tensorflow, por isso não estão respondidas.
'''
model.fit(xs,ys,epochs=400)