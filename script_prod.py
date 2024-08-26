import pandas as pd
import joblib
import os
import datetime

# Definir el directorio donde se encuentran los modelos
model_dir = "/modelos"

# Definir el directorio para guardar los resultados
output_dir = "/resultados"
os.makedirs(output_dir, exist_ok=True)

# Obtener la fecha actual
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Cargar el modelo DecisionTree entrenado usando la fecha actual
dt_model_filename = f'{model_dir}/decision_tree_model_{current_date}.pkl'
dt_model = joblib.load(dt_model_filename)

# Cargar el modelo XGBoost entrenado usando la fecha actual
xgb_model_filename = f'{model_dir}/xgb_best_model_{current_date}.pkl'
xgb_model = joblib.load(xgb_model_filename)

# Leer el archivo test.csv
test_df = pd.read_csv('data/test.csv')

# Seleccionar las características necesarias excluyendo la columna 'id'
# Asumiendo que la columna 'id' es la primera columna
features = test_df.columns.tolist()
features.remove('id')

# Predecir usando el modelo DecisionTree
dt_predictions = dt_model.predict(test_df[features])
test_df['dt_predictions'] = dt_predictions

# Predecir usando el modelo XGBoost
xgb_predictions = xgb_model.predict(test_df[features])
test_df['xgb_predictions'] = xgb_predictions

# Guardar el DataFrame con las predicciones, incluyendo la columna 'id'
output_filename = f'{output_dir}/predictions_{current_date}.csv'
test_df.to_csv(output_filename, index=False)

print(f"Predicciones guardadas en {output_filename}")
