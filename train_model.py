import pandas as pd
import xgboost as xgb
import pickle


df = pd.read_csv('Agri_yield_prediction.csv') 

# Select features and target (adjust column names to match your CSV)
X = df[['Rainfall', 'Temperature']] 
y = df['Yield']

# Initialize and train the XGBoost Regressor
model = xgb.XGBRegressor()
model.fit(X, y)

# Save the trained model to a file so it can be loaded later
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)