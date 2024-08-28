import pandas as pd
import requests
import json

URL = "http://localhost:8000/predict"

test_data = pd.read_csv('data/test.csv')

def get_prediction(data):
    response = requests.post(URL, json=data)
    return response.json()

for index, row in test_data.iterrows():
    data = row.to_dict()

    try:
        prediction = get_prediction(data)
        print(f"Datos de entrada: {data}")
        print(f"Predicción: {prediction}")
        print("---")
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar la solicitud: {e}")
    except json.JSONDecodeError:
        print("Error al decodificar la respuesta JSON")

print("Pruebas completadas.")