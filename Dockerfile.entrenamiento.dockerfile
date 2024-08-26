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

# Copiar los datos necesarios para el entrenamiento
COPY data/train.csv data/train.csv

# Copiar el script de entrenamiento
COPY script_train.py ./

# Ejecutar el script de entrenamiento
CMD ["python", "script_train.py"]
