import time
import pickle
import glob
from gcloud import storage
from params import *


def save_model(serialized_model : pickle, model_filename: str) -> None:
    """
    Save the model on GoogleCloudStorage
    serialized_model -> the model should be serialized first using 'pickle.dumps(model)'
    model_filename -> the format should be 'file_name.pkl'
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        # Save model on GCS
        client = storage.Client()
        print(client)
        bucket = client.bucket('ml_psy')
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_string(serialized_model)
        return f'✅ Model saved to GCS'

    except Exception:
        return f'❌ An error occured'


def load_model(model_filename : str):
    """
    Download the model from GoogleCloudStorage
    """
    try:
        # Download model from GCS
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(model_filename)
        model_downloaded = blob.download_as_string()
        print(f'✅ Model downloaded from GCS')
        return model_downloaded

    except Exception:
        return f'❌ An error occured'
