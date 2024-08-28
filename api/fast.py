import pandas as pd
from io import BytesIO

from google.cloud import storage
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_logic.registry import load_model

app = FastAPI()

#https://storage.cloud.google.com/ml_psy/data_preprocessed/test_data_main_dataset_X_preprocessed.csv

#Get the data
storage_client = storage.Client()
bucket = storage_client.bucket('ml_psy')
blob = bucket.blob("data_preprocessed/test_data_main_dataset_X_preprocessed.csv")

# Download the blob content (test data preproc) as bytes
content = blob.download_as_bytes()

# Read the bytes into a pandas DataFrame
df = pd.read_csv(BytesIO(content))

#load trained models
model_addictive_disorder = load_model('model_addictive_dis.pkl', 'ml_psy')
model_anxiety_disorder = load_model('model_anxiety_dis.pkl', 'ml_psy')
model_mood_disorder = load_model('model_mood_dis.pkl', 'ml_psy')
model_schizophrenia = load_model('model_schizo.pkl', 'ml_psy')
model_trauma_and_stress = load_model('model_trauma_stress.pkl', 'ml_psy')

# Allowing all middleware (optional but good practice for dev purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Implementing the root endpoint /
@app.get("/")
def root():
    return {'ML Psy': '🔥🔥🔥'}


# Implement the rood predict to get prediction from the imported model
@app.get("/predict")
def predict(patient= 1, threshold = 0.2) -> str:
    """
    Make a single prediction of mental disorder
    """
    #check that the models dowloaded are not empty
    models = [model_addictive_disorder, model_anxiety_disorder , model_mood_disorder, model_schizophrenia, model_trauma_and_stress]
    for model in models :
        assert model is not None

    #choose a patient :
    selected_patient = df.iloc[patient]

    #probabilités de prédiction par modèle
    proba_addictive_disorder = models[0].predict_proba(selected_patient)
    proba_anxiety_disorder = models[1].predict_proba(selected_patient)
    proba_mood_disorder = models[2].predict_proba(selected_patient)
    proba_schizophrenia = models[3].predict_proba(selected_patient)
    proba_trauma_and_stress = models[4].predict_proba(selected_patient)

    #dictionnaire qui contient la probabilité de yes des maladies avec le plus difference entre yes and no
    best_proba = {}

    #selection des maladies avec la plus grosse difference entre yes and no
    if abs(proba_addictive_disorder[0][0] - proba_addictive_disorder[0][1]) > threshold :
        best_proba["addictive disorder"] = proba_addictive_disorder[0][1]
    if abs(proba_anxiety_disorder[0][0] - proba_anxiety_disorder[0][1]) > threshold :
        best_proba["anxiety disorder"] = proba_anxiety_disorder[0][1]
    if abs(proba_mood_disorder[0][0] - proba_mood_disorder[0][1]) > threshold :
        best_proba["mood disorder"] = proba_mood_disorder[0][1]
    if abs(proba_schizophrenia[0][0] - proba_schizophrenia[0][1]) > threshold :
        best_proba["schizophrenia"] = proba_schizophrenia[0][1]
    if abs(proba_trauma_and_stress[0][0] - proba_trauma_and_stress[0][1]) > threshold :
        best_proba["trauma and stress related disorder"] = proba_trauma_and_stress[0][1]


    # tu return un json avec la maladie et sa proba de 1

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON


    return (best_proba)
