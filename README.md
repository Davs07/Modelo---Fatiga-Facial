# Modelo de Fatiga Facial 

Este repositorio contiene notebooks y scripts para el desarrollo y prueba de un modelo de detección de fatiga facial utilizando un dataset descargado de Kaggle.

## Contenido

- `Modelo_Fatiga_Facial.ipynb`: Notebook para configurar y descargar el dataset desde Kaggle, necesario para entrenar el modelo de detección de fatiga facial.
  [![Open In Colab](https://colab.research.google.com/drive/1mJR6gg4PRB6J-4RIRhiHDZGsy1bSRCce?usp=sharing)
- `Prueba_Modelo_Fatiga.ipynb`: Notebook para probar el modelo TFLite de detección de fatiga facial utilizando imágenes de prueba.
- `main.py`: Script que implementa un sistema de detección de fatiga en tiempo real utilizando la cámara web.

## Descripción

Este proyecto tiene como objetivo desarrollar un modelo que pueda detectar signos de fatiga facial en imágenes. Utiliza un dataset de Kaggle para entrenar el modelo y luego lo prueba en imágenes de prueba. Además, se incluye un script para la detección de fatiga en tiempo real.

### Modelo_Fatiga_Facial.ipynb

- Configura las credenciales de Kaggle.
- Descarga y descomprime el dataset `driver-drowsiness-dataset-ddd` desde Kaggle.
- Código para la preparación del dataset y entrenamiento del modelo.

### Prueba_Modelo_Fatiga.ipynb

- Monta Google Drive para acceder a los archivos del modelo.
- Carga y prepara el modelo TFLite.
- Preprocesa imágenes de prueba.
- Realiza predicciones sobre las imágenes para detectar fatiga facial (clasificación binaria: "Drowsy" o "Non-Drowsy").

### main.py

- Captura de video en tiempo real utilizando la cámara web.
- Procesamiento de imágenes para detectar fatiga usando MediaPipe Face Mesh.
- Cálculo de métricas de fatiga como la relación de aspecto del ojo (EAR) y la relación de aspecto de la boca (MAR).
- Generación de alertas cuando se detecta fatiga.
- Registro de datos en un archivo CSV.

## Uso

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Davs07/Modelo---Fatiga-Facial.git
   cd Modelo---Fatiga-Facial

2. Abre los notebooks en Google Colab:
   - [Modelo_Fatiga_Facial.ipynb](https://colab.research.google.com/github/Davs07/Modelo---Fatiga-Facial/blob/main/Modelo_Fatiga_Facial.ipynb)
   - [Prueba_Modelo_Fatiga.ipynb](https://colab.research.google.com/github/Davs07/Modelo---Fatiga-Facial/blob/main/Prueba_Modelo_Fatiga.ipynb)

3. Sigue las instrucciones en los notebooks para configurar las credenciales de Kaggle, descargar el dataset, y realizar el entrenamiento y prueba del modelo.

## Requisitos

- Python 3.10
- TensorFlow
- OpenCV
- Google Colab
- MediaPipe
- SciPy

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request para cualquier mejora o corrección.

