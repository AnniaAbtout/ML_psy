import os
import numpy as np

DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# INSTANCE = os.environ.get("INSTANCE")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'))
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'))

COLUMN_NAMES_RAW = []

DTYPES_RAW = {

}

DTYPES_PROCESSED = np.float32


################## VALIDATIONS #################

env_valid_options = dict(
    DATA_SIZE=[],
    MODEL_TARGET=[],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
