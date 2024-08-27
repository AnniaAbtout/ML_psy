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

    # Get feature names (for the sex category, add manually the name of the column)
    sex_feature_names = ['sex']
    num_feature_names = preprocessor.transformers_[1][1].get_feature_names_out()

    # Combine feature names
    feature_names = list(sex_feature_names) + list(num_feature_names)

    return pd.DataFrame(X_transformed, columns = feature_names)

def preprocess_target(y: pd.DataFrame, disease: str):
    """
    Preprocessing the target to obtain a OneHotEncoded target with 'Healthy' control as 0
    """
    # Defining categories to set 'Healthy contro' as 0 in the OnheHotEncoder
    categories = [['Healthy control', disease]]

    # One Hot Encoder
    ohe_binary = OneHotEncoder(categories=categories, sparse_output=False, drop="if_binary")
    y_encoded = ohe_binary.fit_transform(pd.DataFrame(y))[:,0]

    return y_encoded
