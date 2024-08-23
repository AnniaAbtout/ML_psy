import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import time

def random_forest(X_preprocessed, y_preprocessed, model_choice):

    # RF implementation
    if model_choice == 'randomforest':
        def randomforest(X_preprocessed, y_preprocessed):
            model_rf = RandomForestClassifier(n_estimators=116)
            model_rf.fit(X_preprocessed, y_preprocessed)
            y_pred = random_forest_cl.predict(X_val)

    # prediction of y
    y_pred = model.predict(y_val)

    return print('F1 score =', f1_score(y_val, y_pred, average='weighted'))

def adaboost(X_preprocessed, y_preprocessed):

    # train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y_preprocessed, test_size=0.1, stratify=y_preprocessed)

    # RF implementation
    model = RandomForestClassifier(n_estimators=116)
