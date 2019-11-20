import sys
import time
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

# from xgboost import XGBClassifier

# plt.plot([1,2,3,4])
# plt.show()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

CATEGORICAL_FEATURES = ["E23", "E24", "E25"]
NUMERICAL_FEATURES = []
data = pd.read_csv(r"./train.csv")
label = pd.read_csv(r"./train_label.csv")

data["click"] = label["label"]
print(data.head())
print("平均点击率： ", data["click"].mean())


# print(pd.DataFrame(data).head())

def Data_exploration(data, categorical_features, numerical_features, max_feature=100):
    for feature in categorical_features + numerical_features:
        mean_click = data.groupby(feature)["click"].mean()
        if len(data[feature].unique()) > max_feature:
            continue
        plt.figure(figsize=(10, 3.2))
        plt.subplots_adjust(wspace=0.4)
        plt.subplot(1, 2, 1)
        sns.countplot(x=feature, data=data)
        if len(data[feature].unique()) > 20:
            plt.xticks([])
        elif len(data[feature].unique()) > 10:
            plt.xticks(rotation=45)
        plt.ylabel("count")

        plt.subplot(1, 2, 2)
        plt.ylabel("mean click ratio")
        sns.barplot(x=feature, y="click", data=data)
        if len(data[feature].unique()) > 20:
            plt.xticks([])
        elif len(data[feature].unique()) > 10:
            plt.xticks(rotation=45)

        plt.show()


#
#
#
Data_exploration(data, CATEGORICAL_FEATURES, NUMERICAL_FEATURES)

lr = LogisticRegression()
gbdt=GradientBoostingClassifier()
xgb=XGBClassifier()
lgbm=LGBMClassifier()