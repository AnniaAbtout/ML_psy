from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_selector
import pandas as pd

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline:
    - SimpleImputer and RobustScaler for the numerical values
    - OneHotEncoder for the categorical values
    """

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

    return pd.DataFrame(X_transformed, columns = preprocessor.get_feature_names_out())

def preprocess_target(y, encoded_disease, dico=False):
    # Dictionnary with encoded diseases, to be printed or not
    dict_disease = {0:'Addictive disorder', 1: 'Anxiety disorder', 2: 'Healthy control', 3: 'Mood disorder', 4: 'Obsessive compulsive disorder', 5: 'Schizophrenia', 6: 'Trauma and stress related disorder'}
    if dico == True:
        print(dict_disease)

    # One Hot Encoder
    oe = OneHotEncoder(sparse_output=False)
    y_oe = oe.fit_transform(y)
    y_transformed = y_oe[:,encoded_disease]

    return pd.DataFrame(y_transformed)
