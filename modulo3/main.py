import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.__version__

b2wCorpus = pd.read_csv("modulo3/data/b2w-10k.csv")     # ler o .cs
print(b2wCorpus.head())                                 # mostra as tabelas
print(b2wCorpus["review_text"])                         # mostra APENAS os textos de review

# Pré-processamento
print(b2wCorpus["reviewer_gender"].value_counts())

