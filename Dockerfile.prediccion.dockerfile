FROM python:3.9

WORKDIR /app

# Instalar dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Crear los directorios necesarios
RUN mkdir -p data /modelos /resultados

# Declarar los volúmenes para los modelos entrenados y los resultados
VOLUME /modelos
VOLUME /resultados

# Copiar los datos necesarios para las predicciones
COPY data/test.csv data/test.csv

# Copiar el script de predicciones
COPY script_prod.py ./

# Ejecutar el script de predicciones
CMD ["python", "script_prod.py"]
