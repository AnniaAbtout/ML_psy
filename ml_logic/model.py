import numpy as np
import pandas as pd
from ml_logic.preprocessor import preprocess_features, preprocess_target
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time

def model_addictive_disorder(data : pd.DataFrame):

    #Definition of the mask to filter the database based on the selected disease
    mask = (data['main.disorder'] == 'Addictive disorder') | (data['main.disorder'] == 'Healthy control')

    # Filtering the dataframe on the selected disease
    df_filtered = data[mask]

    # Cleaning the dataframe depending on the target 'main disorder' or 'specific disorder')
    X = df_filtered.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    y = df_filtered['main.disorder']

    # Using preprocessing functions on X and y
    X_preprocessed = preprocess_features(X)
    y_preprocessed = preprocess_target(pd.DataFrame(y), 'Addictive disorder')

    #define the model : random forest classifier
    rf_addictive_disorder = RandomForestClassifier('criterion': 'gini',
                                                    'max_depth': 6,
                                                    'max_features': 'sqrt',
                                                    'n_estimators': 150)

    #train the model
    rf_addictive_disorder.fit(X_preprocessed, y_preprocessed)

    return rf_addictive_disorder

def model_anxiety_disorder(data : pd.DataFrame):

    #Definition of the mask to filter the database based on the selected disease
    mask = (data['main.disorder'] == 'Anxiety disorder') | (data['main.disorder'] == 'Healthy control')

    # Filtering the dataframe on the selected disease
    df_filtered = data[mask]

    # Cleaning the dataframe depending on the target 'main disorder' or 'specific disorder')
    X = df_filtered.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    y = df_filtered['main.disorder']

    # Using preprocessing functions on X and y
    X_preprocessed = preprocess_features(X)
    y_preprocessed = preprocess_target(pd.DataFrame(y), 'Anxiety disorder')

    #define the model : xgboost
    xgb_anxiety_disorder = XGBClassifier(objective='binary:logistic',
                                            eval_metric='logloss',
                                            use_label_encoder=False,
                                            colsample_bytree = 0.9,
                                            learning_rate = 0.01,
                                            max_depth = 7,
                                            n_estimators = 100,
                                            subsample = 0.9)

    #train the model
    xgb_anxiety_disorder.fit(X_preprocessed, y_preprocessed)

    return xgb_anxiety_disorder

def model_mood_disorder(data : pd.DataFrame):

    #Definition of the mask to filter the database based on the selected disease
    mask = (data['main.disorder'] == 'Mood disorder') | (data['main.disorder'] == 'Healthy control')

    # Filtering the dataframe on the selected disease
    df_filtered = data[mask]

    # Cleaning the dataframe depending on the target 'main disorder' or 'specific disorder')
    X = df_filtered.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    y = df_filtered['main.disorder']

    # Using preprocessing functions on X and y
    X_preprocessed = preprocess_features(X)
    y_preprocessed = preprocess_target(pd.DataFrame(y), 'Mood disorder')

    #define the model : xgboost
    xgb_mood_disorder = XGBClassifier(objective='binary:logistic',
                                            eval_metric='logloss',
                                            use_label_encoder=False,
                                            colsample_bytree = 0.8,
                                            learning_rate = 0.1,
                                            max_depth = 7,
                                            n_estimators = 200,
                                            subsample = 0.9)

    #train the model
    xgb_mood_disorder.fit(X_preprocessed, y_preprocessed)

    return xgb_mood_disorder

def model_schizophrenia(data : pd.DataFrame):

    #Definition of the mask to filter the database based on the selected disease
    mask = (data['main.disorder'] == 'Schizophrenia') | (data['main.disorder'] == 'Healthy control')

    # Filtering the dataframe on the selected disease
    df_filtered = data[mask]

    # Cleaning the dataframe depending on the target 'main disorder' or 'specific disorder')
    X = df_filtered.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    y = df_filtered['main.disorder']

    # Using preprocessing functions on X and y
    X_preprocessed = preprocess_features(X)
    y_preprocessed = preprocess_target(pd.DataFrame(y), 'Schizophrenia')

    #define the model : xgBoost
    xgb_schizophrenia = XGBClassifier(objective='binary:logistic',
                                            eval_metric='logloss',
                                            use_label_encoder=False,
                                            colsample_bytree = 0.8,
                                            learning_rate = 0.2,
                                            max_depth = 7,
                                            n_estimators = 200,
                                            subsample = 0.9)

    #train the model
    xgb_schizophrenia.fit(X_preprocessed, y_preprocessed)

    return xgb_schizophrenia

def model_trauma_and_stress_related_disorder(data : pd.DataFrame):

    #Definition of the mask to filter the database based on the selected disease
    mask = (data['main.disorder'] == 'Trauma and stress related disorder') | (data['main.disorder'] == 'Healthy control')

    # Filtering the dataframe on the selected disease
    df_filtered = data[mask]

    # Cleaning the dataframe from the non necessary columns
    X = df_filtered.drop(columns = ['specific.disorder', 'eeg.date', 'Unnamed: 122', 'main.disorder'])
    #define the target
    y = df_filtered['main.disorder']

    # Using preprocessing functions on X and y
    X_preprocessed = preprocess_features(X)
    y_preprocessed = preprocess_target(pd.DataFrame(y), 'Trauma and stress related disorder')

    #define the model : rf
    rf_trauma_and_stress = RandomForestClassifier(criterion =  'entropy',
                                                    max_depth = 5,
                                                    max_features = 'sqrt',
                                                    n_estimators = 200)

    #train the model
    rf_trauma_and_stress.fit(X_preprocessed, y_preprocessed)

    return rf_trauma_and_stress
