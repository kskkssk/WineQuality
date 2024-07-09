import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("winequality-red.csv")

X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

gbm = LGBMClassifier()
pipe_gbm = Pipeline([('scaler', StandardScaler()), ('multi', OneVsRestClassifier(gbm))])
pipe_gbm.fit(X_train, y_train)


log = LogisticRegression()
pipe_log = Pipeline([('scaler', StandardScaler()), ('multi', OneVsRestClassifier(log))])
pipe_log.fit(X_train, y_train)
