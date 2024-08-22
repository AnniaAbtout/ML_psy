import pandas as pd

from sklearn.decomposition import PCA
from ml_logic.preprocessor import preprocess_features

def PCA_eeg_features(X : pd.dataFrame, n_compo=66) --> list :
    #dataframe des EEG features uniquement
    X_eeg = X.drop(columns=["no.","sex","age","eeg.date", "education","IQ","specific.disorder","main.disorder","Unnamed: 122"])

    #preprocessing des eeg features : robustScaler
    X_eeg_preproc = preprocess_features(X_eeg)

    #run the pca on the preprocessed data
    pca = PCA(n_components=n_compo) #number of components is chosen with the help of paper
    X_eeg_proj = pca.fit_transform(X_eeg_preproc)

    #Output une list des variances de chaque PC dans l'ordre en pourcentage
    pca_result = pca.explained_variance_ratio_*100

    return pca_result
