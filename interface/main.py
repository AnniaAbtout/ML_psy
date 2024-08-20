import pandas as pd
import numpy as np

from setup import *
from ml_logic.preprocessor import preprocess_features

# Preprocess
X = df.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])

X_preprocessed = preprocess_features(X)
