import pandas as pd
from sklearn.decomposition import PCA

def PCA_eeg_features(X_preproc : pd.DataFrame, n_compo=66): #--> list & pd.DataFrame
    #dataframe des EEG features uniquement, elles sont déjà préprocessées
    X_eeg_preproc = X_preproc.drop(columns=["cat_transformer__sex_M","num_transformer__age","num_transformer__education","num_transformer__IQ"])

    #run the pca on the preprocessed data
    pca = PCA(n_components=n_compo) #number of components is chosen with the help of paper
    X_eeg_proj = pca.fit_transform(X_eeg_preproc)

    #Output une list des variances de chaque PC dans l'ordre en pourcentage
    pca_result = pca.explained_variance_ratio_*100

    return pca_result, X_eeg_proj
