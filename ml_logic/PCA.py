import pandas as pd
from sklearn.decomposition import PCA

def PCA_eeg_features(X_preproc : pd.dataFrame, n_compo=66): #--> list & pd.DataFrame
    #dataframe des EEG features uniquement, elles sont déjà préprocessées
    X_eeg_preproc = X_preproc.drop(columns=["sex","age","education","IQ"])

    #run the pca on the preprocessed data
    pca = PCA(n_components=n_compo) #number of components is chosen with the help of paper
    X_eeg_proj = pca.fit_transform(X_eeg_preproc)

    #Output une list des variances de chaque PC dans l'ordre en pourcentage
    pca_result = pca.explained_variance_ratio_*100

    return pca_result, X_eeg_proj
