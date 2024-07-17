from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
from models import get_data, train, test
from config import config


mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("LGBM")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()

    gbm = LGBMClassifier(**config["gbm_classification"], random_state=42)
    pipe_gbm = Pipeline([('scaler', StandardScaler()), ('multi', OneVsRestClassifier(gbm))])

    with mlflow.start_run():

        mlflow.log_params(config["gbm_classification"])

        train(pipe_gbm, X_train, y_train)
        test(pipe_gbm, X_test, y_test)

        mlflow.sklearn.log_model(pipe_gbm, "light_gbm_model")