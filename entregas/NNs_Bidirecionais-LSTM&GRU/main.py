# IMPORTS
import tensorflow as tf
from tensorflow import keras
from keras.layers import TextVectorization, Embedding, Bidirectional, GRU, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical # para gerar o y como one-hot, já que é problema multiclasse
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

print(tf.__version__)

usecols = ["review_text", "overall_rating"]
b2wCorpus = pd.read_csv("modulo6/data/b2w-10k.csv", usecols=usecols)
print(b2wCorpus.head())

# PRE-PROCESSAMENTO #

# check overall_rating
b2wCorpus["overall_rating"].unique()
# não tem nenhuma linha cuja nota não é um valor numérico entre 0 e 5


# train, test split
random_state = 42
test_size = 0.25

x_values = b2wCorpus["review_text"]
y_values = b2wCorpus["overall_rating"]

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, random_state=random_state, test_size=test_size)


# EMBEDDING # 

# Exibe as GPUs disponíveis (deve exibir pelo menos uma GPU)
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs disponíveis:", physical_devices)

# Configuração para alocar memória de forma dinâmica
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Cria a camada TextVectorization
output_sequence_length = 10000
vectorizer = TextVectorization()
vectorizer.adapt(x_train) # Adaptar o vetorizador aos dados de treinamento
vocab = vectorizer.get_vocabulary() # Verificar o vocabulário criado pelo vetorizador
vectorized_train_data = vectorizer(x_train).numpy() # Vetorizar os dados de treinamento
vectorized_train_data

print("Vocabulário:", vocab)
print(f"{len(vectorized_train_data[0])}")

'''
    Codificar rótulos de teste no formato one-hot
    Subtraindo 1 dos rótulos para ajustar a indexação
    A função to_categorical espera que os rótulos sejam inteiros começando de 0 até num_classes - 1.
'''
num_classes = b2wCorpus["overall_rating"].nunique()
y_train_encoded = to_categorical(y_train - 1, num_classes=num_classes)
y_test_encoded = to_categorical(y_test - 1, num_classes=num_classes)

# Ajustar ou truncar as sequências nos dados de treinamento e teste para validação
vectorized_train_data = vectorizer(x_train).numpy()
vectorized_test_data = vectorizer(x_test).numpy()

input_len = len(vectorized_train_data[0])
vectorized_train_data_padded = pad_sequences(vectorized_train_data, maxlen=input_len, padding='post')
vectorized_test_data_padded = pad_sequences(vectorized_test_data, maxlen=input_len, padding='post')

# MODELS #

# Criação do modelo
input_dim = len(vocab) + 2
input_len = len(vectorized_train_data[0])
embedding_dim = 50
output_dim = 10
num_classes = b2wCorpus["overall_rating"].nunique()

optimizer = 'adam'
loss = 'categorical_crossentropy'
epochs = 20
metrics = ['AUC']
epochs=30
batch_size=50

# LSTM UNI #
lstm_uni_model = Sequential()
lstm_uni_model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len))

lstm_uni_model.add(LSTM(64, return_sequences=True))  # Usar return_sequences=True para conectar camadas LSTM em sequência
lstm_uni_model.add(Dropout(0.5))
lstm_uni_model.add(LSTM(64, return_sequences=True))
lstm_uni_model.add(Dropout(0.5))
lstm_uni_model.add(LSTM(64))
lstm_uni_model.add(Dropout(0.5))

lstm_uni_model.add(Dense(64, activation='relu', input_dim=input_dim))
lstm_uni_model.add(Dense(num_classes, activation='softmax'))  # Camada de saída com softmax para problemas multiclasse
lstm_uni_model.compile(optimizer=optimizer, loss=loss, metrics=metrics) # Compilar o modelo
lstm_uni_model.fit(vectorized_train_data_padded, y_train_encoded, epochs=epochs, batch_size=batch_size) # treina

# GRU UNI #
gru_uni_model = Sequential()
gru_uni_model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len))

gru_uni_model.add(GRU(64, return_sequences=True))
gru_uni_model.add(Dropout(0.5))
gru_uni_model.add(GRU(64, return_sequences=True))
gru_uni_model.add(Dropout(0.5))
gru_uni_model.add(GRU(64))
gru_uni_model.add(Dropout(0.5))

gru_uni_model.add(Dense(64, activation='relu'))
gru_uni_model.add(Dense(num_classes, activation='softmax')) # camada de saída
gru_uni_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
gru_uni_model.fit(vectorized_train_data_padded, y_train_encoded, epochs=epochs, batch_size=batch_size)

# LSTM BI #
lstm_bi_model = Sequential()
lstm_bi_model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len))

lstm_bi_model.add(Bidirectional(LSTM(64, return_sequences=True)))
lstm_bi_model.add(Dropout(0.5))
lstm_bi_model.add(Bidirectional(LSTM(64, return_sequences=True)))
lstm_bi_model.add(Dropout(0.5))
lstm_bi_model.add(Bidirectional(LSTM(64)))
lstm_bi_model.add(Dropout(0.5))

lstm_bi_model.add(Dense(64, activation='relu'))
lstm_bi_model.add(Dense(num_classes, activation='softmax')) # camada de saída
lstm_bi_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
lstm_bi_model.fit(vectorized_train_data_padded, y_train_encoded, epochs=5, batch_size=batch_size)

# GRU BI #
gru_bi_model = Sequential()
gru_bi_model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_len))

gru_bi_model.add(Bidirectional(GRU(64, return_sequences=True)))
gru_bi_model.add(Dropout(0.5))
gru_bi_model.add(Bidirectional(GRU(64, return_sequences=True)))
gru_bi_model.add(Dropout(0.5))
gru_bi_model.add(Bidirectional(GRU(64)))
gru_bi_model.add(Dropout(0.5))

gru_bi_model.add(Dense(64, activation='relu')) # camada densa
gru_bi_model.add(Dense(num_classes, activation='softmax')) # camada de saída

gru_bi_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
gru_bi_model.fit(vectorized_train_data_padded, y_train_encoded, epochs=5, batch_size=batch_size)

# SUMÁRIO PARA -> LSTM UNI
lstm_uni_model.summary()

# SUMÁRIO PARA -> LSTM BI
lstm_bi_model.summary()

# SUMÁRIO PARA -> GRU UNI
gru_uni_model.summary()

# SUMÁRIO PARA -> GRU BI
gru_bi_model.summary()


# VALIDATION #
lstm_uni_model_loss, lstm_uni_model_accuracy = lstm_uni_model.evaluate(vectorized_test_data_padded, y_test_encoded)
lstm_bi_model_loss, lstm_bi_model_accuracy = lstm_bi_model.evaluate(vectorized_test_data_padded, y_test_encoded)
gru_uni_model_loss, gru_uni_model_accuracy = gru_uni_model.evaluate(vectorized_test_data_padded, y_test_encoded)
gru_bi_model_loss, gru_bi_model_accuracy = gru_bi_model.evaluate(vectorized_test_data_padded, y_test_encoded)

print(f"Perda e Acurácia do modelo LSTM unidirecional nos dados de teste: {lstm_uni_model_loss}, {lstm_uni_model_accuracy}")
print(f"Perda e Acurácia do modelo LSTM bidirecional nos dados de teste: {lstm_bi_model_loss}, {lstm_bi_model_accuracy}")
print(f"Perda e Acurácia do modelo GRU unidirecional nos dados de teste: {gru_uni_model_loss}, {gru_uni_model_accuracy}")
print(f"Perda e Acurácia do modelo GRU bidirecional nos dados de teste: {gru_bi_model_loss}, {gru_bi_model_accuracy}")


# CRIAÇÃO DO .CSV #

# agrupando resultados no df
dict_results = {}
dict_results["Método"] = ["LSTM Unidirecional", "LSTM Bidirecional", "GRU Unidirecional", "GRU Bidirecional"]
dict_results["Acurácia"] = [lstm_uni_model_accuracy, lstm_bi_model_accuracy, gru_uni_model_accuracy, gru_bi_model_accuracy]

df_results = pd.DataFrame(dict_results)

# Adição da coluna "Melhor (S/N)" com base na maior acurácia dos modelos
df_results["Melhor (S/N)"] = ["S" if acc == df_results["Acurácia"].max() else "N" for acc in df_results["Acurácia"]]
df_results = df_results.sort_values(by="Acurácia", ascending=False)
df_results

df_results.to_csv("entregas/data/df_results.csv")
