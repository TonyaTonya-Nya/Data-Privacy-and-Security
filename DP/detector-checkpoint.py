
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn
import matplotlib.pyplot as plt
import pickle
import shap
import random
from sklearn import metrics
import os
import diffprivlib


def readFile(trainPath, labelPath):
    data = pd.read_csv(trainPath, header=None)
    label = pd.read_csv(labelPath, header=None)
    feature = []
    for i in range(data.shape[0]):
        tmp = data.iloc[i, 1:132].values.tolist()
        feature.append(tmp)

    feature = np.array(feature)
    ans = []
    for i in range(label.shape[0]):
        if str(label.iloc[i, 0]) == 'BenignWare':
            ans.append(0)
        else:
            ans.append(1)
    ans = np.array(ans)
    return feature, ans


### load data ###
train_feature, train_label = readFile('../train.csv', '../train_label.csv')
test_feature, test_label = readFile('../test4-2.csv', '../test_label.csv')
model = pickle.load(open('./KNN.pkl', 'rb'))

### shuffle ###
indices = np.arange(train_feature.shape[0])
np.random.shuffle(indices)
train_feature = train_feature[indices]
train_label = train_label[indices]

### 印出 shap ###
#shap.initjs()
#explainer = shap.KernelExplainer(model.predict_proba, train_feature[0:2000])
#shap_values = explainer.shap_values(test_feature[0:2])
#shap.force_plot(explainer.expected_value[0], shap_values[0], test_feature[0])
#shap.summary_plot(shap_values, test_feature[0:2])

### 印出準確率  ###
predict_ans = model.predict(test_feature)
print('accuracy_score :', metrics.accuracy_score(predict_ans, test_label))
print('recall_score   :', metrics.recall_score(
    predict_ans, test_label, average='macro'))
print('precision_score:', metrics.precision_score(
    predict_ans, test_label, average='macro'))


### 印出各類機率 ###
#predict_ans = model.predict_proba(test_feature)


# convert array into dataframe
#DF = pd.DataFrame(predict_ans)

# save the dataframe as a csv file
# DF.to_csv("data1.csv")
