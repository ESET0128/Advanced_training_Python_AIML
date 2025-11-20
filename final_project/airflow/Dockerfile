FROM apache/airflow:2.8.0

USER root
RUN apt-get update && apt-get install -y build-essential

USER airflow

COPY requirements.txt .
RUN pip install --no-cache-dir requests
RUN pip install --no-cache-dir -r requirements.txt
