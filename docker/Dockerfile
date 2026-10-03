# One image for both services. Airflow's official image (Airflow's own pinned
# dependencies) plus the ML libraries; mlflow doubles as the tracking server.
FROM apache/airflow:3.3.2-python3.12

# Writable data dir for the MLflow server (named volume inherits this ownership).
USER root
RUN mkdir -p /mlflow && chown airflow /mlflow
USER airflow

RUN pip install --no-cache-dir \
    "mlflow==3.16.1" \
    "scikit-learn==1.7.2" \
    "pandas==2.3.3" \
    "matplotlib==3.10.7" \
    "joblib" \
    "python-dotenv"
