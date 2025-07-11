import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
import gcsfs # Importar gcsfs

def train_model(data_path: str, model_output_dir: str):
    """
    Trains a linear regression model and saves it to a GCS path.

    Args:
        data_path (str): GCS path to the input CSV data (e.g., "gs://your-bucket/data.csv").
        model_output_dir (str): GCS path to save the trained model (e.g., "gs://your-bucket/models/").
    """
    print(f"Cargando datos desde: {data_path}")
    # Usar pandas para leer directamente desde GCS gracias a gcsfs
    df = pd.read_csv(data_path)
    print("Datos cargados exitosamente.")

    # Preparar los datos
    # Asegúrate de que estas características coincidan con las de tus datos
    EXPECTED_FEATURES = ['bedrooms', 'bathrooms', 'sq_footage']
    target_feature = 'price'

    X = df[EXPECTED_FEATURES]
    y = df[target_feature]

    print("Entrenando el modelo de regresión lineal...")
    model = LinearRegression()
    model.fit(X, y)
    print("Modelo entrenado exitosamente.")

    # Definir la ruta completa del archivo del modelo en GCS
    model_file_name = "model.joblib"
    full_model_gcs_path = os.path.join(model_output_dir, model_file_name)

    print(f"Guardando el modelo en: {full_model_gcs_path}")

    # Crear el sistema de archivos GCS
    fs = gcsfs.GCSFileSystem()

    # Asegurarse de que el directorio exista en GCS
    # gcsfs.mkdirs crea directorios en GCS de forma recursiva
    # No es necesario usar os.makedirs aquí
    fs.mkdirs(model_output_dir, exist_ok=True)

    # Guardar el modelo directamente en GCS usando gcsfs
    # joblib.dump puede escribir directamente a una ruta fsspec
    with fs.open(full_model_gcs_path, 'wb') as f:
        joblib.dump(model, f)

    print("Modelo guardado exitosamente en GCS.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a house price prediction model.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="GCS path to the input CSV data (e.g., 'gs://your-bucket/data.csv')"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="GCS directory to save the trained model (e.g., 'gs://your-bucket/models/')"
    )
    args = parser.parse_args()

    train_model(args.data_path, args.model_dir)
