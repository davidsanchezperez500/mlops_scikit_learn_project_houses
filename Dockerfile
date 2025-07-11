FROM us-central1-docker.pkg.dev/mlops-training-462812/docker-repository/house-price-base:latest

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
