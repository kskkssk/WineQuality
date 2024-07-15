from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from model_train import get_data, load_model, test

with DAG(
    'hw_kudasheva',
    default_args={'depends_on_past': False,
                  'email': ['kudasheva0.kudasheva@gmail.com'],
                  'email_on_failure': True,
                  'email_on_retry': True,
                  'retries': 1},
    start_date=datetime(2024, 6, 1),
    schedule=timedelta(days=1),
        catchup=False) as dag:

    t1 = PythonOperator(task_id='get_data', python_callable=get_data)
    t2 = PythonOperator(task_id='load_model', python_callable=load_model)
    t3 = PythonOperator(task_id='test', python_callable=test)

t1 >> t2 >> t3
