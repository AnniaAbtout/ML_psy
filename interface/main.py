import pandas as pd
import numpy as np

from params import *
from ml_logic.preprocessor import preprocess_features, preprocess_target
from ml_logic.PCA import PCA_eeg_features

# Get the data from our CSV file in raw_data
url = 'raw_data/train_data_main_dataset.csv'
df = pd.read_csv(url)

# Define our target ('main_disorder' or 'specific_disorder')
target = 'main'

# Selection of the disease to be studied
encoded_disease = int(input("""Choose >>
1: Addictive disorder
2: Anxiety disorder
3: Mood disorder
4: Obsessive compulsive disorder
5: Schizophrenia
6: Trauma and stress related disorder
= """))

dict_disease = {1:'Addictive disorder',
                2: 'Anxiety disorder',
                3: 'Mood disorder',
                4: 'Obsessive compulsive disorder',
                5: 'Schizophrenia',
                6: 'Trauma and stress related disorder'
                }

# Definition of the mask to filter the database based on the selected disease
mask = (df['main.disorder'] == dict_disease[encoded_disease]) | (df['main.disorder'] == 'Healthy control')

# Filtering the dataframe on the selected disease
df_filtered = df[mask]

# Cleaning the dataframe depending on the target 'main disorder' or 'specific disorder')
if target == 'specific':
    X = df_filtered.drop(columns = ['main.disorder', 'eeg.date', 'Unnamed: 122', 'specific.disorder'])
    y = df_filtered['specific.disorder']
else:
    X = df_filtered.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    y = df_filtered['main.disorder']

# Using preprocessing functions on X and y
X_preprocessed = preprocess_features(X)
y_preprocessed = preprocess_target(pd.DataFrame(y), dict_disease[encoded_disease])

#run PCA on eeg features only
X_preprocessed_pca = PCA_eeg_features(X_preprocessed, n_compo=100) #dataframe contenant les différentes PCs, nombre de PC=n_compo

#update the train dataFrame X with PCs and personnal features (sex, education, age, IQ)
X_preprocessed_pca[["age","sex","education","IQ"]] = X_preprocessed[["age","sex","education","IQ"]]
