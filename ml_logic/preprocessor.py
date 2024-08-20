from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_selector
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/thomasbergeron/code/AnniaAbtout/ML_psy/raw_data/EEG.machinelearing_data_BRMH.csv')

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline:
    - SimpleImputer and RobustScaler for the numerical values
    - OneHotEncoder for the categorical values
    """

    X = df.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    print(X)

    # Data split
    num_col = make_column_selector(dtype_include=['float64'])
    cat_col = make_column_selector(dtype_include=['O'])

    # Impute then scale numerical values:
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('standard_scaler', RobustScaler())])

    # Encode categorical values
    cat_transformer = OneHotEncoder(handle_unknown='ignore', drop='if_binary')

    # Parallelize "num_transformer" and "cat_transfomer"
    preprocessor = ColumnTransformer([
        ('cat_transformer', cat_transformer, cat_col),
        ('num_transformer', num_transformer, num_col)])

    X_transformed = preprocessor.fit_transform(X)
    #print(X_transformed)

    return pd.DataFrame(X_transformed, columns = preprocessor.get_feature_names_out())


print(preprocess_features(df))
