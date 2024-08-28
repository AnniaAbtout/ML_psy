import pandas as pd
import numpy as np

from params import *
from ml_logic.model import model_addictive_disorder, model_anxiety_disorder, model_mood_disorder, model_schizophrenia, model_trauma_and_stress_related_disorder
from ml_logic.registry import save_model

# Get the data from our CSV file in raw_data
url = 'raw_data/train_data_main_dataset.csv'
df = pd.read_csv(url)

#preprocess and train each specialized model
model_addictive_disorder = model_addictive_disorder(df)
model_anxiety_disorder = model_anxiety_disorder(df)
model_mood_disorder = model_mood_disorder(df)
model_schizophrenia = model_schizophrenia(df)
model_trauma_and_stress = model_trauma_and_stress_related_disorder(df)

#save the model in google cloud
save_model(model_addictive_disorder, 'model_addictive_dis.pkl')
save_model(model_anxiety_disorder, 'model_anxiety_dis.pkl')
save_model(model_mood_disorder, 'model_mood_dis.pkl')
save_model(model_schizophrenia, 'model_schizo.pkl')
save_model(model_trauma_and_stress, 'model_trauma_stress.pkl')
