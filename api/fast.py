import pandas as pd
from io import BytesIO

from google.cloud import storage
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from ml_logic.registry import load_model
# from ml_logic.preprocessor import preprocess_features

app = FastAPI()

#Get the data
storage_client = storage.Client()
bucket = storage_client.bucket('ml_psy')
blob = bucket.blob('models/model.pkl')

# Download the blob content as bytes
content = blob.download_as_bytes()

# Read the bytes into a pandas DataFrame
df = pd.read_csv(BytesIO(content))

#load the model
model_downloaded = blob.download_as_string()
model = pickle.loads(model_downloaded)

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
def predict(patient= df.sample(1)) -> str:
    """
    Make a single prediction of mental disorder
    """
    assert model is not None
    
    # prendre ton patient x dans ton x_process
    # faire passser ta donner dans tes 8 modeles deja loadés
    y_pred = model.predict(patient)
    
    # tu return un json avec la maladie et sa proba de 1 
    
    # X_pred = pd.DataFrame(locals(), index=[0])

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return (y_pred)