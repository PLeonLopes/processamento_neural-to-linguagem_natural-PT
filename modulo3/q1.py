import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.__version__

b2wCorpus = pd.read_csv("modulo3/data/b2w-10k.csv")     # ler o .csv

# A)
print(b2wCorpus["review_text"])                         # mostra os textos de review
print(b2wCorpus["recommend_to_a_friend"])               # mostra recomendação 0 || 1

# B)
b2wCorpus["recommend_to_a_friend"] = b2wCorpus["recommend_to_a_friend"].map({"Yes": 1, "No": 0})    # Converter a coluna "recommend_to_a_friend" de str para int

# Mostrar as colunas para verificar a conversão
print(b2wCorpus["review_text"])
print(b2wCorpus["recommend_to_a_friend"])