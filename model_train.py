import pandas as pd
import subprocess
import joblib
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
    subprocess.run(['dvc', 'pull'])
    with open('model.pkl', 'rb') as f:
        model = joblib.load(f)
    return model


def test():
    from sklearn.metrics import f1_score
    from datetime import datetime
    model = train()
    _, X_test, _, y_test = get_data()
    y_pred = model.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    return f1


def update_model():
    from datetime import datetime
    current_f1 = test()
    subprocess.run(['dvc', 'pull', 'metrics'])
    try:
        with open('metrics.json', 'r') as f:
            current_metrics = json.load(f)
            if current_metrics['f1'] < current_f1:
                subprocess.run(['dvc', 'add', 'metrics.json'])
                subprocess.run(['dvc', 'push'])
                with open('metrics.json', 'w') as f:
                    json.dump({'f1': current_f1, 'timestamp': datetime.now()}, f)
                    print(f"Last score {current_metrics['f1']}")
                    print(f"Updated model with f1 score: {current_f1}")
    except:
        subprocess.run(['dvc', 'add', 'metrics'])
        subprocess.run(['dvc', 'push'])


