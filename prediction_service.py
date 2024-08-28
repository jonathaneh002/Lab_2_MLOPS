import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Mobile Price Range Predictor")

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "directorios_locales", "modelos")
results_dir = os.path.join(base_dir, "directorios_locales", "resultados")

os.makedirs(results_dir, exist_ok=True)

dt_model_path = os.path.join(model_dir, "decision_tree_model_2024-08-25.pkl")
xgb_model_path = os.path.join(model_dir, "xgb_best_model_2024-08-25.pkl")

dt_model = joblib.load(dt_model_path)
try:
    xgb_model = joblib.load(xgb_model_path)
except Exception as e:
    print(f"Error al cargar el modelo XGBoost: {str(e)}")
    print("El servicio continuará sin el modelo XGBoost.")
    xgb_model = None

class MobileFeatures(BaseModel):
    battery_power: int
    clock_speed: float
    fc: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    blue: int
    dual_sim: int
    four_g: int
    three_g: int
    touch_screen: int
    wifi: int

class PredictionResponse(BaseModel):
    dt_prediction: int
    xgb_prediction: int | None

def save_prediction_to_csv(features: dict, dt_pred: int, xgb_pred: int | None):
    csv_path = os.path.join(results_dir, "predictions.csv")
    
    result = {**features, "dt_prediction": dt_pred, "xgb_prediction": xgb_pred}
    
    df = pd.DataFrame([result])
    
    
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: MobileFeatures):
    try:
        df = pd.DataFrame([features.dict()])
    
        dt_pred = int(dt_model.predict(df)[0])
        
        xgb_pred = None
        if xgb_model is not None:
            xgb_pred = int(xgb_model.predict(df)[0])
        
        save_prediction_to_csv(features.dict(), dt_pred, xgb_pred)
        
        return PredictionResponse(dt_prediction=dt_pred, xgb_prediction=xgb_pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Mobile Price Range Predictor API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)