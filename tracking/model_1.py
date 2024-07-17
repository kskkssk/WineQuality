from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import mlflow
from models import get_data, train, test
from config import config


mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("LogReg")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()

    log = LogisticRegression(max_iter=1000, **config["logistic_regression"], random_state=42)
    pipe_log = Pipeline([('scaler', StandardScaler()), ('multi', OneVsRestClassifier(log))])

    with mlflow.start_run():

        mlflow.log_params(config['logistic_regression'])

        train(pipe_log, X_train, y_train)
        test(pipe_log, X_test, y_test)

        mlflow.sklearn.log_model(pipe_log, "logistic_regression_model")