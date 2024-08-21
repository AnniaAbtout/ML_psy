import pandas as pd
import numpy as np

from params import *
from ml_logic.preprocessor import preprocess_features, preprocess_traget

df = pd.read_csv('/Users/thomasbergeron/code/AnniaAbtout/ML_psy/raw_data/train_data_main_dataset.csv')

# Preprocess
target = 'specific'

if target == 'specific':
    X = df.drop(columns = ['main.disorder', 'eeg.date', 'Unnamed: 122', 'specific.disorder'])
    y = df['specific.disorder']
else:
    X = df.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    y = df['main.disorder']


X_preprocessed = preprocess_features(X)
y_preprocessed = preprocess_traget(y)

print(y_preprocessed)
