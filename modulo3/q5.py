import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization

tf.__version__

b2wCorpus = pd.read_csv("modulo3/data/b2w-10k.csv")     # Ler o arquivo CSV

# Selecionar as colunas relevantes e converter "recommend_to_a_friend" de str para int
b2wCorpus["recommend_to_a_friend"] = b2wCorpus["recommend_to_a_friend"].map({"Yes": 1, "No": 0})

# Separar features e labels
X = b2wCorpus["review_text"]
y = b2wCorpus["recommend_to_a_friend"]

y = y.astype(int)       # checagem de tipo para INT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)       # Dividir em conjuntos de treino e teste

# Conversão de labels para np.array de float32
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# Exibir as GPUs disponíveis
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs disponíveis:", physical_devices)

# Configuração para alocar memória de forma dinâmica (c?)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Amostragem dos dados
plt.hist([len(linha.split()) for linha in X_train])
plt.xlabel('Comprimento do Review')
plt.ylabel('Frequência')
plt.show()

max_length = 100        # Definição do comprimento máximo das sequências com base no histograma

# Criação da camada TextVectorization
vectorizer = TextVectorization(
    max_tokens=20000,       # Número máximo de tokens no vocabulário
    output_mode='int',      # tipo do output
    output_sequence_length=max_length       # max_length
)

vectorizer.adapt(X_train)       # Adaptação do vetorizador aos dados de treinamento

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
embedding_dim = 128  # Dimensão dos vetores de embedding
input_len = vectorized_train_data.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(vocab),
        output_dim=embedding_dim,
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

# Treinar o modelo
history = model.fit(vectorized_train_data, y_train, epochs=20, validation_split=0.2, batch_size=32)

# Avaliar o modelo
loss, accuracy = model.evaluate(vectorized_test_data, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Plotar a perda (loss)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar a acurácia (accuracy)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
