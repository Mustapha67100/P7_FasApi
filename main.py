# 1. Library imports
import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib
import json
import os
import shap
from pathlib import Path

# 2. Create the app object
app = FastAPI()

# Repertpoire de chargement des données

#CURRENT_FOLDER = Path.cwd()
CURRENT_FOLDER= os.getcwd()
PROJECT_FOLDER = Path(CURRENT_FOLDER)
DATA_FOLDER = PROJECT_FOLDER
model=joblib.load(DATA_FOLDER/'model.joblib')
#model=joblib.load(DATA_FOLDER/'model_log.joblib')

#model=joblib.load(DATA_FOLDER/'model_log_100000.joblib')
data_test=joblib.load(DATA_FOLDER/'data_test_sub_cutoff.joblib')
shap_values=joblib.load(DATA_FOLDER/'shap_values.joblib')

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a client_id, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
  
# Requête permettant de récupérer les informations d’un client sélectionné à travers son identifiant ID
    
@app.get('/predict/{client_id}')
def get_client_id(client_id:int):
  data_client_id= data_test.loc[client_id]
  print(data_client_id)
  # convertir les données en dataframe 
  data_client_id_df=pd.DataFrame(data_client_id)
  print(data_client_id_df)
  # Transposer le dataframe
  data_client_id_df= data_client_id_df.T
  print(data_client_id_df)
  proba = model.predict_proba(data_client_id_df)
  
  return dict(
   # FastAPI convertit les dict en json automatiquement
   client_id=client_id,
   proba = proba.tolist(),   
)

  
  #return(json.dumps(proba.tolist()))
  
    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)