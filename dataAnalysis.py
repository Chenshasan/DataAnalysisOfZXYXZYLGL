import sys
import re
import time
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

# from xgboost import XGBClassifier

# plt.plot([1,2,3,4])
# plt.show()
# 暂时没有冗余数据
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

CATEGORICAL_FEATURES=["day","hour","A1","B1","B3","D1","D2","E2","E3","E4","E5","E6","E7","E8","E9","E10","E11","E12","E13","E15","E16","E17","E18","E19","E21","E22","E23","E24","E25","E26","E27","E28","E29"]
NUMERICAL_FEATURES=["A2","A3","B2","C1","C2","C3","E1","E14","E20"]
data=pd.read_csv(r"./train.csv")
label=pd.read_csv(r"./train_label.csv")

data["click"]=label["label"]
# print(data["date"])
data["day"]=data["date"].apply(lambda x:re.split(r"-|\s+|:",x)[2])
data["hour"]=data["date"].apply(lambda x:re.split(r"-|\s+|:",x)[3])

# print(data["day"])
# print(data["hour"].head())
# print("平均点击率： ",data["click"].mean())
# print(pd.DataFrame(data).head())

def Data_exploration(data,categorical_features,numerical_features,max_feature=100):
    for feature in categorical_features+numerical_features:
        mean_click=data.groupby(feature)["click"].mean()
        if len(data[feature].unique())>max_feature:
            print(feature+"  is numerical")
            continue
        plt.figure(figsize=(10,3.2))
        plt.subplots_adjust(wspace=0.4)
        plt.subplot(1,2,1)
        sns.countplot(x=feature,data=data)
        if len(data[feature].unique())>20:
            plt.xticks([])
        elif len(data[feature].unique())>10:
            plt.xticks(rotation=45)
        plt.ylabel("count")

        plt.subplot(1,2,2)
        plt.ylabel("mean click ratio")
        sns.barplot(x=feature,y="click",data=data)
        if len(data[feature].unique())>20:
            plt.xticks([])
        elif len(data[feature].unique())>10:
            plt.xticks(rotation=45)
        plt.show()
Data_exploration(data,CATEGORICAL_FEATURES,NUMERICAL_FEATURES)


"""
Step4 :创建模型
"""
lr = LogisticRegression()
gbdt=GradientBoostingClassifier()
xgb=XGBClassifier()
lgbm=LGBMClassifier()
