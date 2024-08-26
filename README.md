# Lab_2_MLOPS

Este trabajo permite entrenar un modelo de machine learning y generar predicciones utilizando contenedores Docker. Todo el flujo de trabajo, desde el entrenamiento hasta la predicción, está automatizado para ejecutarse de manera eficiente y reproducible.

## Estructura del Proyecto

```
mi_proyecto/
│
├── Dockerfile.entrenamiento      # Dockerfile para construir la imagen de entrenamiento
├── Dockerfile.prediccion         # Dockerfile para construir la imagen de predicción
├── requirements.txt              # Dependencias necesarias para el proyecto
├── script_train.py               # Script de Python para entrenar el modelo
├── script_prod.py                # Script de Python para hacer predicciones
│
├── data/
│   ├── train.csv                 # Conjunto de datos para entrenar el modelo
│   └── test.csv                  # Conjunto de datos para generar predicciones
│
└── directorios_locales/
    ├── modelos/                  # Directorio donde se guardarán los modelos entrenados
    └── resultados/               # Directorio donde se guardarán los resultados de las predicciones
```

## Requisitos Previos

- Docker instalado en tu sistema.
- Conexión a internet para descargar las dependencias de Python.

## Configuración y Ejecución

### 1. Construir la Imagen de Entrenamiento

Desde el directorio raíz del proyecto (`mi_proyecto/`), ejecuta el siguiente comando para construir la imagen Docker de entrenamiento:

```bash
docker build -t entrenamiento-imagen -f Dockerfile.entrenamiento .
```

### 2. Ejecutar el Contenedor de Entrenamiento

Una vez construida la imagen, ejecuta el contenedor para entrenar el modelo. Este comando también mapea los volúmenes locales para que los modelos y resultados se guarden en tu sistema:

```bash
docker run -v ${pwd}/directorios_locales/modelos:/modelos -v ${pwd}/directorios_locales/resultados:/resultados entrenamiento-imagen
```

### 3. Construir la Imagen de Predicción

Después de entrenar el modelo, construye la imagen Docker para realizar predicciones:

```bash
docker build -t prediccion-imagen -f Dockerfile.prediccion .
```

### 4. Ejecutar el Contenedor de Predicción

Ejecuta el contenedor de predicción para generar las predicciones basadas en el modelo entrenado:

```bash
docker run -v ${pwd}/directorios_locales/modelos:/modelos -v ${pwd}/directorios_locales/resultados:/resultados -v ${pwd}/data:/data prediccion-imagen
```

### 5. Resultados

- **Modelos entrenados**: Se guardarán en `mi_proyecto/directorios_locales/modelos/`.
- **Resultados de predicciones**: Se guardarán en `mi_proyecto/directorios_locales/resultados/`.

## Automatización de Tareas

Puedes programar la ejecución de estos contenedores utilizando `cron` en sistemas Unix/Linux o el Programador de Tareas en Windows para automatizar el entrenamiento y las predicciones en horarios específicos, como a la 1 a.m.

## Notas Adicionales

- Asegúrate de que los archivos `train.csv` y `test.csv` estén correctamente ubicados en la carpeta `data/` antes de ejecutar los contenedores.
- Puedes modificar los scripts `script_train.py` y `script_prod.py` según tus necesidades para ajustar el flujo de trabajo.
