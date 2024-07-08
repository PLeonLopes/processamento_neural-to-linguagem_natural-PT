import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

b2wCorpus = pd.read_csv("modulo6/data/b2w-10k.csv")     # lê os csv

print(b2wCorpus.head())             
print(b2wCorpus["review_text"])     # pega só os reviews-text