from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


with DAG(
    'hw_kudasheva',
    default_args={'depends_on_past': False,
                  'email': ['kudasheva0.kudasheva@gmail.com'],
                  'email_on_failure': True,
                  'email_on_retry': True,
                  'retries': 1,
                  'retry_delay': timedelta(days=1)},
    start_date=datetime(2024, 6, 1),
    catchup=False) as dag:

    t1 = PythonOperator(task_id='get_data', python_callable=get_data)
    t2 = PythonOperator(task_id='train', python_callable=train)
    t3 = PythonOperator(task_id='test', python_callable=test)

t1 >> t2 >> t3