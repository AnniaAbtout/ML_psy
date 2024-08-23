import pandas as pd
import numpy as np

from params import *
from ml_logic.preprocessor import preprocess_features, preprocess_target
from ml_logic.PCA import PCA_eeg_features

url = '/Users/thomasbergeron/code/AnniaAbtout/ML_psy/raw_data/train_data_main_dataset.csv'
df = pd.read_csv(url)

# Preprocess the entire dataset
target = 'main'

if target == 'specific':
    X = df.drop(columns = ['main.disorder', 'eeg.date', 'Unnamed: 122', 'specific.disorder'])
    y = df['specific.disorder']
else:
    X = df.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    y = df['main.disorder']

encoded_disease = int(input("Choose >> = {0:'Addictive disorder', 1: 'Anxiety disorder', 2: 'Healthy control', 3: 'Mood disorder', 4: 'Obsessive compulsive disorder', 5: 'Schizophrenia', 6: 'Trauma and stress related disorder'} = "))

X_preprocessed = preprocess_features(X)
y_preprocessed = preprocess_target(pd.DataFrame(y), encoded_disease)
print(y_preprocessed)
#run PCA on eeg features only
X_preprocessed_pca = PCA_eeg_features(X_preprocessed, n_compo=100) #dataframe contenant les différentes PCs, nombre de PC=n_compo

#update the train dataFrame X with PCs and personnal features (sex, education, age, IQ)
X_preprocessed_pca[["age","sex","education","IQ"]] = X_preprocessed[["age","sex","education","IQ"]]
