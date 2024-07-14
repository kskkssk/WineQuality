import pandas as pd
import subprocess
import joblib
import dvc
import json


def get_data():
    from sklearn.model_selection import train_test_split
    subprocess.run(['dvc', 'pull'])
    df = pd.read_csv("winequality-red.csv")
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test


def train():
    from config import config
    X_train, y_train = get_data()
    subprocess.run(['dvc', 'pull'])
    with open('model.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

def metrics_exist(metrics):
    try:
        dvc.api.read('metrics.json', remote='s3_mlops')
        return json.loads(metrics)
    except:
        return None

def test():
    from sklearn.metrics import accuracy_score, f1_score
    from datetime import datetime
    model = train()
    X_test, y_test = get_data()
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')

    metrics = {'f1': f1,
               'accuracy': accuracy,
               'timestamp': datetime.now()}

    if not metrics_exist(metrics):
        subprocess.run(['dvc', 'add', 'metrics'])
    else:
        current_metrics = metrics
        current_metrics['f1'] < metrics['f1'] or current_metrics['accuracy'] < metrics['accuracy']:
        subprocess.run(['dvc', 'add', 'metrics'])
        subprocess.run(['dvc', 'push'])

