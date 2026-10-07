FROM python:3.11-slim-trixie

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    tensorflow==2.21.0 \
    tfx==1.21.0 \
    pandas \
    numpy \
    scikit-learn \
    joblib

COPY . /app

ENV PYTHONPATH=/app

CMD ["python", "src/pipeline/run_pipeline.py"]