import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization, Embedding, Bidirectional, GRU, LSTM, Dense

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

# Vetorização
vectorized_train_data = vectorizer(x_train).numpy()

# Criação do modelo
input_dim = len(vocab) + 2  # Adicionar 2 para <OOV> e <PAD>
embedding_dim = 50
input_len = len(vectorized_train_data[0])
output_dim = 10

model = keras.Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len))
model.add(Bidirectional(GRU(64)))  # Adapte o número de unidades GRU conforme necessário
model.add(Dense(1, activation='sigmoid'))

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(vectorizer(x_train).numpy(), y_train, epochs=5, batch_size=32)

# Avaliar o modelo nos dados de teste
vectorized_test_data = vectorizer(x_test).numpy()
accuracy = model.evaluate(vectorized_test_data, y_test)[1]
print(f"Acurácia do modelo nos dados de teste: {accuracy}")