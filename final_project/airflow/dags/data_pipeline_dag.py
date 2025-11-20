from datetime import datetime, timedelta
from airflow import DAG

from airflow.operators.python import PythonOperator

import pandas as pd
from sqlalchemy import create_engine, text
import os
import requests
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
import os

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data.injestion import DataIngestion


# DB_CONFIG = {
#     "host": "postgres",
#     "database": "meter_db",
#     "user": "postgres",
#     "password": "admin",
#     "port": "5432",
#     "version": "15"
# }
DB_CONFIG = {
    "host": "postgres",
    "database": "meter_db",
    "user": "postgres",
    "password": "admin",
    "port": "5432",
    "version": "15"
}

DATA_FILE_PATH = r"/opt/airflow/data/raw/meter_data.csv"
MODEL_DIR = r"/tmp/models" # Use /tmp for reliable write access
os.makedirs(MODEL_DIR, exist_ok=True)


def run_ingestion():
    ingestion = DataIngestion(DB_CONFIG)
    ingestion.create_table_if_not_exists()
    ingestion.run_pipeline(DATA_FILE_PATH)


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}



# ---------------------------------------------------------
# 2. EXTRACT TASK  (works only with your 3 columns)
# ---------------------------------------------------------
def extract_data(**kwargs):
    ti = kwargs["ti"]

    # CSV that your ingestion pipeline used
    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError("CSV file not found for extraction.")

    csv_df = pd.read_csv(DATA_FILE_PATH, parse_dates=["timestamp"])

    # Read the same table where ingestion inserted values
    engine_url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    engine = create_engine(engine_url)

    db_df = pd.read_sql("SELECT * FROM raw_meter_data", engine)

    # Push to XCom in JSON format
    ti.xcom_push("csv_df", csv_df.to_json(orient="records"))
    ti.xcom_push("db_df", db_df.to_json(orient="records"))


# ---------------------------------------------------------
# 3. TRANSFORM TASK
# ---------------------------------------------------------
def transform_data(**kwargs):
    ti = kwargs["ti"]

    csv_json = ti.xcom_pull(task_ids="extract_data", key="csv_df")
    db_json = ti.xcom_pull(task_ids="extract_data", key="db_df")

    csv_df = pd.read_json(csv_json)
    db_df = pd.read_json(db_json)

    # Combine CSV + DB records
    unified_df = pd.concat([csv_df, db_df], ignore_index=True)

    # Clean datatypes
    unified_df["timestamp"] = pd.to_datetime(unified_df["timestamp"])
    unified_df["load_value"] = unified_df["load_value"].astype(float)

    ti.xcom_push("unified_df", unified_df.to_json(orient="records"))


# ---------------------------------------------------------
# 4. LOAD TASK
# ---------------------------------------------------------
def load_data(**kwargs):
    ti = kwargs["ti"]
    unified_json = ti.xcom_pull(task_ids="transform_data", key="unified_df")

    df = pd.read_json(unified_json)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    engine_url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    engine = create_engine(engine_url)

    # Create final unified table
    create_sql = """
        CREATE TABLE IF NOT EXISTS unified_meter_data (
            meter_id TEXT,
            timestamp TIMESTAMP,
            load_value FLOAT
        );
    """

    with engine.begin() as conn:
        conn.execute(text(create_sql))
        df.to_sql("unified_meter_data", conn, if_exists="append", index=False)


def train_forecast_model(**context):
    ti = context.get("ti")

    # -------------------------------------------------
    # DB CONNECTION
    # -------------------------------------------------
    engine_url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    engine = create_engine(engine_url)

    df = pd.read_sql(
        "SELECT timestamp, load_value FROM unified_meter_data ORDER BY timestamp",
        engine,
    )

    # -------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------
    df["t"] = np.arange(len(df))
    X = df[["t"]]
    y = df["load_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )


    # mlflow.set_tracking_uri("http://mlflow:5000")  # MLflow container
    # mlflow.set_experiment("meter_forecasting")

    # Enable MLflow auto logging
    # mlflow.sklearn.autolog()

    # =================================================
    # RUN MLFLOW TASK
    # =================================================


    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Explicit metric logging
    # mlflow.log_metric("MAE", mae)
    # mlflow.log_metric("RMSE", rmse)

    # Save model manually + log to MLflow
    model_path = os.path.join(MODEL_DIR, "forecast.pkl")
    pickle.dump(model, open(model_path, "wb"))
    # mlflow.log_artifact(model_path)

    # Save metrics JSON + log to MLflow
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    json.dump({"MAE": mae, "RMSE": rmse}, open(metrics_path, "w"), indent=4)
    # mlflow.log_artifact(metrics_path)

    # Push for XCom
    ti.xcom_push("model_path", model_path)
    ti.xcom_push("metrics_path", metrics_path)





with DAG(
    dag_id="data_ingestion_pipeline",
    default_args=default_args,
    description="Pipeline to ingest raw meter data into PostgreSQL",
    start_date=datetime(2025, 11, 18),
    schedule="* * * * *",   
    catchup=False,
    tags=["data", "ingestion", "pipeline"]
) as dag:

    ingest_task = PythonOperator(
        task_id="run_data_ingestion",
        python_callable=run_ingestion
    )

    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data
    )

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )

    train_task = PythonOperator(
        task_id="train_forecast",
        python_callable=train_forecast_model
    )
   
    
    # extract_task >> transform_task >> load_task >> train_task
    ingest_task >> extract_task >> transform_task >> load_task >> train_task 



    
