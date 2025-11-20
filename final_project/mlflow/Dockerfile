FROM python:3.12

# Set a working directory inside the container
WORKDIR /mlflow

# Install MLflow and Postgres driver
RUN pip install --no-cache-dir mlflow psycopg2-binary scikit-learn

# Environment variables (can be overridden by docker-compose)
ENV MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://postgres:admin@postgres:5432/meter_db
ENV MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

# Make sure artifact directory exists inside the image
RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

# Use shell form so ${VAR} expansion works
CMD mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000
