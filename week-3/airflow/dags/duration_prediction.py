#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator, get_current_context
from datetime import datetime

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)



def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print(f"Number of records: {len(df)}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    print(f"Number of records after function: {len(df)}")

    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()

    x_train = dv.fit_transform(train_dicts)

    train_model(df, x_train)




def train_model(df, x_train):

    with mlflow.start_run():
        target = 'duration'

        y_train = df[target].values

        lr = LinearRegression()
        lr.fit(x_train,y_train)

        y_pred = lr.predict(x_train)

        rmse = root_mean_squared_error(y_train,y_pred)
        mlflow.log_metric("rmse", rmse)

        intercept = lr.intercept_
        mlflow.log_metric("intercept", intercept)
        print(f"Intercept: {intercept}")

        mlflow.sklearn.log_model(lr, artifact_path="model")




def run():
    
    context = get_current_context()
    execution_date = context["execution_date"]

    month = execution_date.month
    year = execution_date.year
    print(f"Year: {year}, Month: {month}")

    df = read_dataframe(year=year, month=month)



with DAG(
    dag_id="train_nyc_taxi_model",
    start_date=datetime(2021,1,1),
    schedule="@monthly",
    catchup=False
) as dag:
    
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=run
    )