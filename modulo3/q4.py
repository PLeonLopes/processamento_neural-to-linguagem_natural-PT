import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization

tf.__version__

# Ler o arquivo CSV
b2wCorpus = pd.read_csv("modulo3/data/b2w-10k.csv")

# Selecionar as colunas relevantes e converter "recommend_to_a_friend" de str para int
b2wCorpus["recommend_to_a_friend"] = b2wCorpus["recommend_to_a_friend"].map({"Yes": 1, "No": 0})

# Separar features e labels
X = b2wCorpus["review_text"]
y = b2wCorpus["recommend_to_a_friend"]

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exibir as GPUs disponíveis
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs disponíveis:", physical_devices)

# Configuração para alocar memória de forma dinâmica
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

plt.hist([len(linha.split()) for linha in X_train])
plt.xlabel('Comprimento do Review')
plt.ylabel('Frequência')
plt.show()

# Definir o comprimento máximo das sequências com base no histograma
max_length = 100  # Você pode ajustar este valor com base no histograma

# Criar a camada TextVectorization
vectorizer = TextVectorization(
    max_tokens=20000,  # Número máximo de tokens no vocabulário
    output_mode='int',
    output_sequence_length=max_length
)

# Adaptar o vetorizador aos dados de treinamento
vectorizer.adapt(X_train)

# Vetorizar os dados de treinamento
vectorized_train_data = vectorizer(X_train)

# Vetorizar os dados de teste
vectorized_test_data = vectorizer(X_test)

# Verificar as formas das matrizes resultantes
print(f"Forma dos dados de treino vetorizados: {vectorized_train_data.shape}")
print(f"Forma dos dados de teste vetorizados: {vectorized_test_data.shape}")

# Verificar o vocabulário criado pelo vetorizador
vocab = vectorizer.get_vocabulary()
print("Vocabulário:", vocab[:10])  # Mostrar os 10 primeiros tokens do vocabulário

# Criação do modelo
input_len = len(vectorized_train_data[0])
output_dim = 10

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(vocab),
        output_dim=output_dim,
        input_length=input_len
    ),

    # Conv1D + global max pooling
    tf.keras.layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3),
    tf.keras.layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3),
    tf.keras.layers.GlobalMaxPooling1D(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Compilar o modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
