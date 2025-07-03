# MLOps: Predicción de precios de casas con scikit-learn + Vertex AI - training

Este repositorio muestra un flujo completo de MLOps para la predicción de precios de casas utilizando scikit-learn y Vertex AI en Google Cloud Platform. Aquí encontrarás:

- Código y scripts para entrenar un modelo de machine learning localmente y en Vertex AI.
- Instrucciones para preparar el entorno, instalar dependencias y autenticarse con GCP.
- Ejemplos para subir datos a Google Cloud Storage y gestionar artefactos.
- Guía para construir y subir imágenes Docker personalizadas.
- Comandos para ejecutar entrenamientos en Vertex AI, subir modelos y desplegarlos en endpoints para predicción online.
- Archivos de infraestructura como código (Terraform) para aprovisionar recursos en GCP (ver carpeta `mlops_training_gcp_admin`).

El objetivo es mostrar buenas prácticas de MLOps, desde el desarrollo local hasta el despliegue y la inferencia en la nube, facilitando la reproducibilidad y automatización del ciclo de vida de modelos de machine learning.

---
## TODO

- Reentrenamiento Automatizado: Implementar triggers para reentrenar el modelo (ej., en un cronograma, o cuando la calidad del modelo disminuya).
- Monitorización del Modelo: Configurar alertas y paneles en Vertex AI para monitorizar el rendimiento del modelo en producción (deriva de datos, deriva de conceptos, sesgo, etc.).
- Pruebas Automatizadas: Añadir más tests a tu pipeline de GitHub Actions, como pruebas unitarias para tu código de entrenamiento o pruebas de integración después del despliegue.
- Versionado del Modelo: Gestionar las diferentes versiones de tus modelos en el registro de modelos y cómo se despliegan.
- A/B Testing de Modelos: Si tienes múltiples versiones de modelos, puedes usar el tráfico dividido en los endpoints de Vertex AI para dirigir un porcentaje del tráfico a diferentes modelos.


## Requisitos
- Python 3.8+
- GCP SDK configurado y autenticado
- Crear bucket en GCS para guardar modelos

```bash
gcloud auth login
gcloud auth application-default login --project mlops-training-462812
gcloud config set project mlops-training-462812 
```


## Entrenamiento local
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python trainer/train.py  # Entrena el modelo y lo guarda en model/model.joblib
```

## Subir los datos a GCS
```bash
VERTEX_BUCKET=mlops-training-models-for-training-46281
gsutil cp data/housing.csv gs://$VERTEX_BUCKET/data/housing.csv
```

## Construir imagen Docker
```bash
PROJECT_ID=mlops-training-462812
REPO_NAME=docker-repository
IMAGE_NAME=house-price-trainer:latest
REGION=us-central1 

gcloud auth configure-docker ${REGION}-docker.pkg.dev
podman build --platform linux/amd64 -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME} .
podman push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}
```
## Ejecutar entrenamiento en Vertex AI
```bash

gcloud ai custom-jobs create \
  --display-name="House-Price-Training-Job" \
  --region=${REGION} \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME} \
  --args=--data-path=gs://$VERTEX_BUCKET/data/housing.csv,--model-dir=/gcs/$VERTEX_BUCKET/models/house-price-model
```
