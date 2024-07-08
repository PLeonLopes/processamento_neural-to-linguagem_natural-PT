import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Mostrar as dimensões dos conjuntos resultantes
print(f"Dimensões do conjunto de treino: {x_train.shape}, {y_train.shape}")
print(f"Dimensões do conjunto de teste: {x_test.shape}, {y_test.shape}")