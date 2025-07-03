# trainer/train.py (MODIFICADO)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler # Nuevo import
from sklearn.pipeline import Pipeline           # Nuevo import
import joblib
import os
import argparse
import numpy as np

print(f"[INFO] numpy version: {np.__version__}")
print(f"[INFO] pandas version: {pd.__version__}")
try:
    import sklearn
    print(f"[INFO] scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("[INFO] scikit-learn not installed")
print(f"[INFO] joblib version: {joblib.__version__}")

# Lista de características que el modelo REALMENTE espera y utilizará
# Asegúrate de que esta lista coincida con tu EXPECTED_FEATURES en predictor.py
# Y con los nombres de las columnas en tu CSV.
FEATURES = ['bedrooms', 'bathrooms', 'sq_footage']
TARGET = 'price'

def train_model(data_path, model_output_dir):
    """
    Entrena un modelo de regresión lineal con preprocesamiento (StandardScaler)
    encapsulado en un Pipeline y lo guarda.

    Args:
        data_path (str): Ruta al archivo CSV del dataset (puede ser local o GCS).
        model_output_dir (str): Directorio donde se guardará el modelo entrenado (puede ser local o GCS).
    """
    print(f"Cargando datos desde: {data_path}")
    try:
        # Asegúrate de tener 'gcsfs' y 'fsspec' en requirements.txt para leer de gs://
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERROR: No se pudo cargar el archivo CSV desde {data_path}. Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Preparar datos: Seleccionar solo las características y el target
    X = df[FEATURES]
    y = df[TARGET]

    # Crear el pipeline de preprocesamiento y modelo
    # El escalador se entrena y aplica automáticamente dentro del pipeline
    model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),       # Paso 1: Escalar las características
        ('regressor', LinearRegression())   # Paso 2: El modelo de regresión lineal
    ])

    print("Entrenando el pipeline completo (escalador + modelo)...")
    model_pipeline.fit(X, y)
    print("Pipeline entrenado exitosamente.")

    # Asegura que el directorio de salida exista
    os.makedirs(model_output_dir, exist_ok=True)

    # Guardar el pipeline completo (no solo el regressor)
    model_path = os.path.join(model_output_dir, 'model.joblib')
    print(f"Intentando guardar el pipeline en: {model_path}")
    try:
        joblib.dump(model_pipeline, model_path, protocol=4)
        print(f"Pipeline guardado exitosamente en: {model_path}")
    except Exception as e:
        print(f"ERROR: No se pudo guardar el pipeline en {model_path}. Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        default=os.getenv('AIP_MODEL_DIR', './local_model_output'),
        help='El directorio donde guardar el modelo entrenado.')
    parser.add_argument(
        '--data-path',
        dest='data_path',
        type=str,
        default='data.csv',
        help='Ruta al archivo CSV del dataset de entrenamiento.')

    args = parser.parse_args()
    train_model(args.data_path, args.model_dir)
