"""
# MLOps: Predicción de precios de casas con scikit-learn + Vertex AI

## Requisitos
- Python 3.8+
- GCP SDK configurado y autenticado
- Crear bucket en GCS para guardar modelos

## Entrenamiento local
```bash
python run_local.py
```

## Subir modelo al bucket
```bash
gsutil cp model/model.joblib gs://<YOUR_BUCKET>/model.joblib
```

## Subir modelo a Vertex AI
```bash
gcloud ai models upload \
  --region=us-central1 \
  --display-name=house-price-model \
  --artifact-uri=gs://<YOUR_BUCKET>/model.joblib \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest
```

## Crear endpoint
```bash
gcloud ai endpoints create \
  --region=us-central1 \
  --display-name=house-price-endpoint
```

## Desplegar modelo
```bash
gcloud ai endpoints deploy-model <ENDPOINT_ID> \
  --region=us-central1 \
  --model=<MODEL_ID> \
  --display-name=house-price-deployment \
  --machine-type=n1-standard-2
```

## Inferencia
```bash
gcloud ai endpoints predict <ENDPOINT_ID> \
  --region=us-central1 \
  --json-request-body=deploy/predict_schema.json
```
"""
