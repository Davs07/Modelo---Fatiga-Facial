# Modelo de Fatiga Facial 

Este repositorio contiene notebooks para el desarrollo y prueba de un modelo de detección de fatiga facial utilizando un dataset descargado de Kaggle.

## Contenido

- `Modelo_Fatiga_Facial.ipynb`: Notebook para configurar y descargar el dataset desde Kaggle, necesario para entrenar el modelo de detección de fatiga facial.
- `Prueba_Modelo_Fatiga.ipynb`: Notebook para probar el modelo TFLite de detección de fatiga facial utilizando imágenes de prueba.

## Descripción

Este proyecto tiene como objetivo desarrollar un modelo que pueda detectar signos de fatiga facial en imágenes. Utiliza un dataset de Kaggle para entrenar el modelo y luego lo prueba en imágenes de prueba.

### Modelo_Fatiga_Facial.ipynb

- Configura las credenciales de Kaggle.
- Descarga y descomprime el dataset `driver-drowsiness-dataset-ddd` desde Kaggle.
- Código para la preparación del dataset y entrenamiento del modelo.

### Prueba_Modelo_Fatiga.ipynb

- Monta Google Drive para acceder a los archivos del modelo.
- Carga y prepara el modelo TFLite.
- Preprocesa imágenes de prueba.
- Realiza predicciones sobre las imágenes para detectar fatiga facial (clasificación binaria: "Drowsy" o "Non-Drowsy").

## Uso

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Davs07/Modelo---Fatiga-Facial.git
   cd Modelo---Fatiga-Facial
   ```

2. Abre los notebooks en Google Colab:
   - [Modelo_Fatiga_Facial.ipynb](https://colab.research.google.com/github/Davs07/Modelo---Fatiga-Facial/blob/main/Modelo_Fatiga_Facial.ipynb)
   - [Prueba_Modelo_Fatiga.ipynb](https://colab.research.google.com/github/Davs07/Modelo---Fatiga-Facial/blob/main/Prueba_Modelo_Fatiga.ipynb)

3. Sigue las instrucciones en los notebooks para configurar las credenciales de Kaggle, descargar el dataset, y realizar el entrenamiento y prueba del modelo.

## Requisitos

- Python 3.10
- TensorFlow
- OpenCV
- Google Colab

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request para cualquier mejora o corrección.

