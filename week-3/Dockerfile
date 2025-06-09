FROM apache/airflow:2.8.1-python3.10

# Switch to root to install OS and Python dependencies
USER root

# Install system dependencies required by xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python libraries
RUN pip install --no-cache-dir xgboost mlflow
