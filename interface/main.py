import pandas as pd
import numpy as np

from setup import *
from ml_logic.preprocessor import preprocess_features

def create_prepro(df):
    return preprocess_features(df)
