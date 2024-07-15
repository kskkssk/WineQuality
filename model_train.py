import pandas as pd
import subprocess
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datetime import datetime


def get_data():
    subprocess.run(['dvc', 'pull'])
    df = pd.read_csv("winequality-red.csv")
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    data = {"X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}
    return data


def load_model():
    subprocess.run(['dvc', 'pull'])
    with open('model.pkl', 'rb') as f:
        model = joblib.load(f)
    return model


def train():
    model = load_model()
    data = get_data()
    trained_model = model.fit(data["X_train"], data["y_train"])
    return trained_model


def test():
    trained_model = train()
    data = get_data()
    y_pred = trained_model.predict(data['X_test'])
    f1 = f1_score(y_true=data['y_test'], y_pred=y_pred, average='macro')
    return f1


def update_model():
    current_f1 = test()
    trained_model = train()
    subprocess.run(['dvc', 'pull', 'metrics'])
    try:
        with open('metrics.json', 'r') as f:
            current_metrics = json.load(f)
    except FileNotFoundError:
        current_metrics = {'f1': 0}

    if current_metrics.get('f1', 0) < current_f1:
        with open('metrics.json', 'w') as file:
            json.dump({'f1': current_f1, 'timestamp': datetime.now()}, file)

        joblib.dump(trained_model, 'model.pkl')

        subprocess.run(['dvc', 'add', 'metrics.json'])
        subprocess.run(['dvc', 'add', 'model.pkl'])
        subprocess.run(['dvc', 'push'])
        print(f"Last score {current_metrics.get('f1', 'N/A')}")
        print(f"Updated model with f1 score: {current_f1}")
