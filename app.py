# app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from typing import List, Dict, Any
import time # Nuevo import para los reintentos

# ==============================================================================
# IMPORTANTE: AJUSTA ESTA LISTA
# Define explícitamente las características que tu modelo espera,
# EN EL ORDEN CORRECTO en que fueron entrenadas.
# ==============================================================================
EXPECTED_FEATURES = [
    'bedrooms',
    'bathrooms',
    'sq_footage'
    # Asegúrate de que esta lista coincida exactamente con tu FEATURES en trainer/train.py
]

# Inicializa la aplicación Flask
app = Flask(__name__)

# ==============================================================================
# Función para cargar el modelo con reintentos
# ==============================================================================
def load_model_pipeline_with_retries(max_retries=5, initial_delay=5):
    """
    Carga el pipeline de scikit-learn entrenado desde la ruta local con reintentos.
    Vertex AI monta los artefactos de tu modelo en /mnt/models/ por defecto
    dentro del contenedor personalizado.
    """
    # AIP_MODEL_DIR es una variable de entorno proporcionada por Vertex AI
    # que apunta al directorio donde se montan los artefactos del modelo.
    # Usamos '/mnt/models/' como fallback si la variable no está presente,
    # ya que es la ubicación estándar de montaje en Vertex AI.
    model_path = os.environ.get("AIP_MODEL_DIR", "/mnt/models/")
    
    # Construye la ruta completa al archivo del modelo
    full_model_path = os.path.join(model_path, 'model.joblib')
    
    print(f"Intentando cargar el modelo desde: {full_model_path}")

    for i in range(max_retries):
        try:
            # Carga el pipeline completo (preprocesador + modelo)
            model = joblib.load(full_model_path)
            print("Modelo cargado exitosamente.")
            return model
        except FileNotFoundError:
            print(f"Intento {i+1}/{max_retries}: Archivo del modelo no encontrado en {full_model_path}. Reintentando en {initial_delay * (2**i)} segundos...")
            time.sleep(initial_delay * (2**i)) # Espera exponencial
        except Exception as e:
            print(f"Intento {i+1}/{max_retries}: Error al cargar el modelo desde {full_model_path}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(initial_delay * (2**i)) # Espera exponencial
    
    # Si todos los reintentos fallan
    raise RuntimeError(f"Fallo al cargar el modelo después de {max_retries} reintentos desde {full_model_path}.")

# ==============================================================================
# Cargar el modelo al inicio del módulo
# Esto asegura que el modelo se cargue una vez por cada worker de Gunicorn
# cuando el módulo app.py es importado.
# ==============================================================================
MODEL_PIPELINE = load_model_pipeline_with_retries()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para realizar predicciones.
    Espera un cuerpo JSON con una lista de instancias.
    """
    # El modelo ya debería estar cargado; si no lo está, algo fue mal al inicio.
    if MODEL_PIPELINE is None:
        print("ERROR: MODEL_PIPELINE no está cargado en el endpoint /predict.")
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Obtener los datos de entrada del JSON de la solicitud
        instances: List[Dict[str, Any]] = request.get_json(force=True)
        
        # 1. Convertir la lista de diccionarios de entrada a un DataFrame
        input_df = pd.DataFrame(instances)

        # 2. Hacer el preprocesamiento robusto:
        # A. Manejar características faltantes: Añadir columnas que el modelo espera.
        for feature in EXPECTED_FEATURES:
            if feature not in input_df.columns:
                input_df[feature] = pd.NA 

        # B. Filtrar características extra y asegurar el orden correcto:
        input_df = input_df[EXPECTED_FEATURES]

        # 3. Realizar la predicción usando el pipeline completo
        predictions = MODEL_PIPELINE.predict(input_df)

        # 4. Convertir las predicciones a una lista para la respuesta JSON
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud para verificar que el servidor está activo y el modelo cargado."""
    if MODEL_PIPELINE is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False, "reason": "Model not loaded"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

