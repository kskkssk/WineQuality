import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import mlflow
from config import config


def get_data():
    df = pd.read_csv("winequality-red.csv")

    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test


def train(model, X_train, y_train):
    return model.fit(X_train, y_train)


def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1-score', f1)
    classes = model.classes_

    matrix = confusion_matrix(y_test, y_pred)
    matrix_df = pd.DataFrame(matrix, columns=classes)
    matrix_df.to_csv('confusion_matrix.csv')

    mlflow.log_artifact('confusion_matrix.csv', 'confusion_matrix')

    for i, c in enumerate(classes):
        df_aux = pd.DataFrame({
            'class': [1 if y == c else 0 for y in y_test],
            'prob': y_probs[:, i]
        })
        fpr, tpr, _ = roc_curve(df_aux['class'], df_aux['prob'])
        roc_auc = auc(fpr, tpr)
        mlflow.log_metric(f'ROC_AUC_{c}', roc_auc)


