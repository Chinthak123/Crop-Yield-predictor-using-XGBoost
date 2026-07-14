import pandas as pd
import xgboost as xgb
import pickle


df = pd.read_csv('Agri_yield_prediction.csv') 
X = df[['Rainfall', 'Temperature']] 
y = df['Yield']
model = xgb.XGBRegressor()
model.fit(X, y)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
