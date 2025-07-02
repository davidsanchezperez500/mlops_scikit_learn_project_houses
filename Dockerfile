# Dockerfile
FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python", "train.py"]
