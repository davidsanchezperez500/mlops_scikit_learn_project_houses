# trainer/train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
import argparse

def train_model(data_path, model_output_dir):
    """
    Entrena un modelo de regresión lineal simple y lo guarda.

    Args:
        data_path (str): Ruta al archivo CSV del dataset (puede ser local o GCS).
        model_output_dir (str): Directorio donde se guardará el modelo entrenado (puede ser local o GCS).
    """
    print(f"Cargando datos desde: {data_path}")
    try:
        # pandas puede leer directamente de gs:// si google-cloud-storage está instalado
        # y si se ejecuta en un entorno donde gs:// paths son accesibles (como Vertex AI o local con gsutil configurado)
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERROR: No se pudo cargar el archivo CSV desde {data_path}. Error: {e}")
        # Para la ejecución local, asegúrate de que 'data.csv' está en el directorio correcto
        # o que la ruta 'gs://' se maneje correctamente (requiere la librería google-cloud-storage)
        exit(1) # Terminar el script si los datos no se cargan

    features = ['bedrooms', 'bathrooms', 'sq_footage']
    target = 'price'

    X = df[features]
    y = df[target]

    print("Entrenando el modelo de regresión lineal...")
    model = LinearRegression()
    model.fit(X, y)
    print("Modelo entrenado exitosamente.")

    # Asegura que el directorio de salida exista
    # Esto funciona para rutas locales y para rutas GCS (a través del FUSE mount en Vertex AI)
    os.makedirs(model_output_dir, exist_ok=True)

    # Guardar el modelo
    model_path = os.path.join(model_output_dir, 'model.joblib')
    print(f"Intentando guardar el modelo en: {model_path}")
    try:
        joblib.dump(model, model_path, protocol=4)
        print(f"Modelo guardado exitosamente en: {model_path}")
    except Exception as e:
        print(f"ERROR: No se pudo guardar el modelo en {model_path}. Error: {e}")
        # Imprimir el traceback completo para depuración
        import traceback
        traceback.print_exc()
        exit(1) # Terminar el script si el modelo no se guarda


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # AIP_MODEL_DIR es una variable de entorno proporcionada por Vertex AI para la ruta de salida del modelo
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        default=os.getenv('AIP_MODEL_DIR', './local_model_output'), # Valor por defecto para pruebas locales
        help='El directorio donde guardar el modelo entrenado.')
    parser.add_argument(
        '--data-path',
        dest='data_path',
        type=str,
        default='data.csv', # Valor por defecto para pruebas locales
        help='Ruta al archivo CSV del dataset de entrenamiento.')

    args = parser.parse_args()
    train_model(args.data_path, args.model_dir)
