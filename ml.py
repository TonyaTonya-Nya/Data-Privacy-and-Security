from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import math
from collections import Counter
from nltk.util import ngrams
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
import pandas as pd


def get_data():
    df = pd.read_csv('./in.csv')
    return df


df = get_data()  # 取資料
# print(df.head()) # 檢查資料


y = df[['Brent']].values.reshape(-1, 1)  # 取預測目標
# print(y)


X = df.drop('date', axis=1).drop('Brent', axis=1).values  # 取特徵值(濾除特徵值以外的)
# print(X)

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2)  # 資料以8:2做訓練與自我驗證


# KNN
time_start = time.time()
knn = neighbors.NearestCentroid()
knn.fit(X_train, y_train)
y_predict = knn.predict(X_train)
print(accuracy_score(y_train, y_predict))
time_end = time.time()
time_c = time_end - time_start
print('time cost', time_c, 's')
