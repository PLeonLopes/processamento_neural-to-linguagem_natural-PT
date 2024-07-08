import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization

print(tf.__version__)

b2wCorpus = pd.read_csv("modulo6/data/b2w-10k.csv")     # lê os csv

# PRE-PROCESSAMENTO
b2wCorpus_copy = b2wCorpus[["review_text", "recommend_to_a_friend"]].copy()
b2wCorpus_copy["recommend_to_a_friend"] = b2wCorpus_copy["recommend_to_a_friend"].map({"Yes": 1, "No":0})

# SEPARAÇÃO
random_state = 42
test_size = 0.25

x_values = b2wCorpus_copy["review_text"]
y_values = b2wCorpus_copy["recommend_to_a_friend"]
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, random_state=random_state, test_size=test_size)

# Exibir as GPUs disponíveis
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs disponíveis:", physical_devices)

# Configuração para alocar memória de forma dinâmica
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Criar a camada TextVectorization
output_sequence_length = 10000
vectorizer = TextVectorization(output_mode='tf-idf', )

# Adaptar o vetorizador aos dados de treinamento
vectorizer.adapt(x_train)

# Verificar o vocabulário criado pelo vetorizador
vocab = vectorizer.get_vocabulary()
# print("Vocabulário:", vocab)

# Vetorizar os dados de treinamento
vectorized_train_data = vectorizer(x_train).numpy()