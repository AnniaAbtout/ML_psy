import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import storage
from ml_logic.registry import load_model
from ml_logic.preprocessor import preprocess_features
import pickle

app = FastAPI()

#take a x_test preprocossed from a bucket
#load the eight models

#tu vas prendre ton preprocess
#tu vas prendre tes 8 modeles

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
def predict(patient) -> str:
    """
    Make a single prediction of mental disorder
    """
    
    # prendre ton patient x dans ton x_proceese
    # faire passser ta donner dans tes 8 modeles deja loader
    
    # tu return un json avec la maladie et sa proba de 1 
    
    
    
    # X_pred = pd.DataFrame(locals(), index=[0])
    # X_pred = preprocess_features

    # #model = app.state.model
    # blob = bucket.blob('best_model_2.pkl')
    # model_downloaded = blob.download_as_string()
    model = pickle.loads(load_model)
    assert model is not None

    X_processed = preprocess_features(X_test)
    y_pred = model.predict(X_processed)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return (y_pred)