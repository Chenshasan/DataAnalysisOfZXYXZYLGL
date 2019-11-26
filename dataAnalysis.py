import sys
import time
import re
import numpy as np
import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt
import seaborn as sns
# from sympy.physics.continuum_mechanics.beam import matplotlib

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings('ignore')

TRAIN_DATA_PATH = "./train.csv"
TEST_DATA_PATH = "./train_label.csv"


def Get_data(data_path):
    """
    :param data_path: 数据文件路径
    :return:

    """
    data = pd.read_csv(data_path, sep="\t")
    return data


train_data = Get_data(TRAIN_DATA_PATH)

# 分类特征
CATEGORICAL_FEATURES = ["city", "province", "carrier", "devtype", "make", "model", "nnt", "os", "osv",
                        "os_name", "adid", "advert_id", "orderid", "advert_industry_inner",
                        'campaign_id', 'creative_id', 'creative_tp_dnf', 'app_cate_id', 'f_channel',
                        'app_id', 'inner_slot_id', 'creative_type', 'creative_is_jump',
                        'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                        'creative_has_deeplink', 'app_paid', 'advert_name']
# 数值特征
NUMERICAL_FEATURES = ["creative_width", "creative_height"]


def Data_exploration(data, categorical_features, numerical_features, max_feature=100):
    """
    :param data: 数据
    :param categorical_features: 离散特征
    :param numerical_features: 数值特征
    :return:不同特征和点击率、样本量的图

    """
    sns.set()

    print("mean click rate: ", data["click"].mean())
    """
    遍历所有特征(对于离散特征只选取特征值总数小于max_feature默认是100的特征)，画出于不同特征值点击率和样本量的图

    """
    for feature in categorical_features + numerical_features:
        mean_click = data.groupby(feature)["click"].mean()
        if feature in categorical_features:
            if len(data[feature].unique()) > max_feature:
                continue
        plt.figure(figsize=(10, 3.2))

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

# 应该分析训练集和测试集里面各个特征的特征值的包含关系，来确定是分类特征还是数值特征？
# 运行时会有一些警告，嫌烦可以关掉
# 加载数据
train_data=pd.read_csv(r"./train.csv")
label=pd.read_csv(r"./train_label.csv")
test_data=pd.read_csv(r"./test.csv")

# 数据预处理：没有空值和无效行；取train_label.csv里的点击结果加在train_data后面
train_data["click"]=label["label"]

#删除指定的字段
feat_Remv = ["ID"]
train_data = train_data.drop(feat_Remv, axis = 1)

user_ids=test_data["ID"]
test_data = test_data.drop(feat_Remv, axis = 1)

#训练集和测试集各自有多少行
train_num, test_num = train_data.shape[0], test_data.shape[0]

#特征和标签
X = pd.concat([train_data.drop("click", axis = 1), test_data])
train_y = train_data["click"]
print(X)
print(train_y)
"""
    Step 3: 特征工程

"""
CATEGORICAL_FEATURES=["day","hour","A1","B1","B3","D1","D2","E2","E3","E4","E5","E6","E7","E8","E9","E10","E11","E12","E13","E15","E16","E17","E18","E19","E21","E22","E23","E24","E25","E26","E27","E28","E29"]
NUMERICAL_FEATURES=["A2","A3","B2","C1","C2","C3","E1","E14","E20"]

def Feature_engineering(X, y, train_num, skb_samples=10000):
    """
    :param X: 特征
    :param y: 标签
    :param train_num: 数据划分行索引节点
    :param skb_samples: 抽样样本数
    :return: 特征工程结果
    """


    x = X.copy()

    print("create day and hour")
    # date拆分成day和hour；
    x["day"] = x["date"].apply(lambda x: re.split(r"-|\s+|:", x)[2])
    x["hour"] = x["date"].apply(lambda x: re.split(r"-|\s+|:", x)[3])
    del x["date"]
    print("#" * 50)

    print("label encoder categorical features")
    # 类别标签数值化
    categorical_features = CATEGORICAL_FEATURES
    for feature in categorical_features:
        label_encoder = preprocessing.LabelEncoder()
        x[feature] = label_encoder.fit_transform(x[feature].astype(str))
    print("#" * 50)

    print("standard scaler numerical features")
    # 数值特征标准化
    standard_scaler = preprocessing.StandardScaler()
    numerical_features = NUMERICAL_FEATURES
    x[numerical_features] = standard_scaler.fit(x[numerical_features]).transform(x[numerical_features])
    print("#" * 50)

    # 特征选择
    print("select features")
    train_x, test_x = x.iloc[:train_num, :], x.iloc[train_num:, :]   #todo:原来的代码到这里会报错，一定记得用iloc和loc
    skb = SelectKBest(k=(100 if 100 <= x.shape[1] else x.shape[1]))
    skb.fit(train_x[:skb_samples], y[:skb_samples])
    support_index = [index for index in range(skb.get_support().shape[0]) if skb.get_support()[index]]
    train_x, test_x = train_x.iloc[:, support_index], test_x.iloc[:, support_index]
    print("#" * 50)

    return train_x, test_x

skb_samples = 1000
train_x,test_x = Feature_engineering(X, train_y, train_num, skb_samples)


"""
Step 4: 创建模型

"""
lr = LogisticRegression()
gbdt = GradientBoostingClassifier()
xgb = XGBClassifier()
lgbm = LGBMClassifier()


def ModelOptimization(model, params, train_x, train_y):
    """
    :param model: 模型
    :param params: 参数
    :param train_x: 训练数据特征
    :param train_y: 训练数据标签
    :return: 最优模型参数

    """
    best_params = []
    for param in params:
        cv = GridSearchCV(estimator=model,  # 模型
                          param_grid=param,  # 参数列表
                          scoring="neg_log_loss",  # 评分规则
                          cv=3,  # 交叉验证次数
                          n_jobs=-1,  #
                          )
        cv.fit(train_x, train_y)
        best_params.append(cv.best_params_)
    return best_params

#cv_samples = 100000
cv_samples=1000
#不同模型常见参数组合
lr_params = [{"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10]},
             {"class_weight": [None, "balanced"]},
             {"penalty": ["l1", "l2"]},
             {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]

lr_best_params = ModelOptimization(lr, lr_params, train_x[:cv_samples], train_y[:cv_samples])
print("lr_best_params: ", lr_best_params)

gbdt_params = [{"n_estimators": [100, 300, 500, 1000]}, #
               {"learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
               {"max_features": [None, "log2", "sqrt"]},
               {"max_depth": [3, 5, 7, 9]},
               {"min_samples_split": [2, 4, 6, 8]},
               {"min_samples_leaf": [1, 3, 5, 7]}]
gbdt_best_params = ModelOptimization(gbdt, gbdt_params, train_x[:cv_samples], train_y[:cv_samples])
print("gbdt_best_params: ", gbdt_best_params)


xgb_params = [{"learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]},
              {"n_estimators": [100, 300, 500, 1000]},
              {"max_depth": range(3,10,2)},
              {"min_child_weight": range(1,6,2)},
              {"gamma": [i/10.0 for i in range(0,5)]},
              {"subsample": [i/10.0 for i in range(6,10)]},
              {"colsample_bytree": [i/10.0 for i in range(6,10)]},
              {"reg_alpha": [1e-5, 1e-2, 0.1, 1, 100]}]
xgb_best_params = ModelOptimization(xgb, xgb_params, train_x[:cv_samples], train_y[:cv_samples])
print("xgb_best_params: ", xgb_best_params)

lgbm_params = [{"learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1]},
               {"num_leaves": [10, 20, 30, 40, 50]},
               {"max_depth": [1, 3, 5, 7, 9]},
               {"n_estimators": [100, 300, 500]}]
lgbm_best_params= ModelOptimization(lgbm, lgbm_params, train_x[:cv_samples], train_y[:cv_samples])
print("lgbm_best_params: ", lgbm_best_params)

"""
    Step 6:模型评估

"""
def Model_evaluation(model, train_x, train_y):
    score = -cross_val_score(model,
                             train_x,
                             train_y,
                             scoring = "neg_log_loss",
                             cv = 3
                            ).mean()
    return score

# evaluation_samples = 100000
evaluation_samples = 1000

best_lr = LogisticRegression(C=3.0,
                             class_weight=None,
                             max_iter=100,
                             penalty="l2",
                             solver="newton-cg")

best_gbdt = GradientBoostingClassifier(n_estimators=300,
                                       learning_rate=0.3,
                                       max_features=None,
                                       max_depth=5,
                                       min_samples_split=8,
                                       min_samples_leaf=3)

best_xgb = XGBClassifier(learning_rate=0.3,
                         n_estimators=500,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0.3,
                         subsample=0.6,
                         colsample_bytree=0.8,
                         reg_alpha=1)

best_lgbm = LGBMClassifier(learning_rate=0.1,
                           num_leaves=20,
                           max_depth=7,
                           n_estimators=100)

lr_score = Model_evaluation(best_lr, train_x[:evaluation_samples], train_y[:evaluation_samples])
gbdt_score = Model_evaluation(best_gbdt, train_x[:evaluation_samples], train_y[:evaluation_samples])
xgb_score = Model_evaluation(best_xgb, train_x[:evaluation_samples], train_y[:evaluation_samples])
lgbm_score = Model_evaluation(best_lgbm, train_x[:evaluation_samples], train_y[:evaluation_samples])

print("lr_score: %.4f" % lr_score)
print("gbdt_score: %.4f" % gbdt_score)
print("xgb_score: %.4f" % xgb_score)
print("lgbm_score: %.4f" % lgbm_score)

"""
    Step 8: Model Ensembling

"""
def Model_ensembling(pred_y_list):
    ensembling_pred_y = np.array(pred_y_list).mean(axis = 0)
    return ensembling_pred_y

# fit_samples = 100000
fit_samples =1000
best_lr = LogisticRegression(C=3.0, class_weight=None, max_iter=100, penalty="l2", solver="newton-cg")
best_gbdt = GradientBoostingClassifier(n_estimators=300, learning_rate=0.3, max_features=None,
                                       max_depth=5, min_samples_split=8, min_samples_leaf=3)
best_xgb = XGBClassifier(learning_rate=0.3, n_estimators=500, max_depth=5, min_child_weight=1, gamma=0.3,
                         subsample=0.6, colsample_bytree=0.8, reg_alpha=1)
best_lgbm = LGBMClassifier(learning_rate=0.1, num_leaves=20, max_depth=7, n_estimators=100)

best_lr.fit(train_x[:fit_samples], train_y[:fit_samples])
lr_pred_y = best_lr.predict_proba(test_x)[:, 1]

best_gbdt.fit(train_x[:fit_samples], train_y[:fit_samples])
gbdt_pred_y = best_gbdt.predict_proba(test_x)[:, 1]

best_xgb.fit(train_x[:fit_samples], train_y[:fit_samples])
xgb_pred_y = best_xgb.predict_proba(test_x)[:, 1]

best_lgbm.fit(train_x[:fit_samples], train_y[:fit_samples])
lgbm_pred_y = best_lgbm.predict_proba(test_x)[:, 1]

pred_y_list = [lr_pred_y, gbdt_pred_y, xgb_pred_y, lgbm_pred_y]

ensembling_pred_y = Model_ensembling(pred_y_list)

"""
    Step 9: Submit

"""


def Submit(user_ids, pred_y):
    submission = pd.DataFrame({"ID": user_ids, "label": pred_y})
    submission.to_csv("./submission.csv", index=False)

# user_ids 是之前的ID
# Submit(user_ids, lr_pred_y)
# Submit(user_ids, gbdt_pred_y)
# Submit(user_ids, xgb_pred_y)
# Submit(user_ids, lgbm_pred_y)
Submit(user_ids, ensembling_pred_y)































# 这一部分在正式处理中不需要
# print(data.head())
# print(data["day"])
# print(data["hour"].head())
# print("平均点击率： ",data["click"].mean())
# print(pd.DataFrame(data).head())
# 暂时没有冗余数据
CATEGORICAL_FEATURES=["day","hour","A1","B1","B3","D1","D2","E2","E3","E4","E5","E6","E7","E8","E9","E10","E11","E12","E13","E15","E16","E17","E18","E19","E21","E22","E23","E24","E25","E26","E27","E28","E29"]
NUMERICAL_FEATURES=["A2","A3","B2","C1","C2","C3","E1","E14","E20"]
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
# Data_exploration(data,CATEGORICAL_FEATURES,NUMERICAL_FEATURES)


# 画图分析数据
# city 城市
# province 省份
Data_exploration(train_data, ["city", "province"], [])

# 画图分析数据
# carrier 运营商
# nnt 联网类型
Data_exploration(train_data, ["carrier", "nnt"], [])

train_data[['model', 'make', "devtype"]].sample(10)

# 对品牌(make)深入分析
train_data["make"] = train_data["make"].fillna("")


def mapMake(make_dict, make):
    """
    :param make_dict: 值映射
    :param make: 值
    :return:
    """
    sign = False
    for brand in make_dict.keys():
        if brand in make:
            sign = True
            return make_dict[brand]
    if ~sign:
        return "others"


# 做指点映射
make_dict = {"iphone": "apple", "iphone7,2": "apple", "iphone9,2": "apple", "iphone8,2": "apple", "iphone8.1": "apple",
             "iphone7,2": "apple",
             "redmi": "xiaomi", "honor": "huawei", "huawei": "huawei", "hisense": "hisense", "letv": "leshi",
             "lemobile": "leshi",
             "lephone": "leshi", "blephone": "leshi", "konka": "konka", "vivo": "vivo", "oppo": "oppo",
             "meizu": "meizu", "samsung": "samsung",
             "htc": "htc", "360": "360", "gionee": "jinli", "jinli": "jinli", "smartisan": "chuizi", "chuizi": "chuizi",
             "koobee": "koobee",
             "apple": "apple", "苹果": "apple", "华为": "huawei", "boway": "boway", "changhong": "changhong",
             "cmdc": "cmdc", "coolpad": "coolpad",
             "nubia": "nubia", "oneplus": "oneplu", "sony": "sony", "doov": "doov", "eva": "eva", "gn": "gn",
             "le": "leshi", "xiaolajiao": "xiaolajiao",
             "小辣椒": "xiaolajiao", "mi": "xiaomi", "cam": "cam", "hm": "hm", "zte": "zte", "bbk": "bbk", "bln": "bln",
             "bnd": "bnd"}

train_data["make"] = train_data["make"].apply(lambda x: x.lower())
train_data["make"] = train_data["make"].apply(lambda x: mapMake(make_dict, x))
Data_exploration(train_data, ['model', 'make', "devtype"], [])

train_data[['os', "os_name", "osv"]].head()

Data_exploration(train_data, ['os', "os_name", "osv"], [])

train_data[["advert_id", "advert_industry_inner"]].head()

Data_exploration(train_data, ['advert_id', "advert_industry_inner"], [])

train_data['time'].head()

# 对于时间信息，我们单独进行统计。把时间信息分为每小时来计算
train_data['hour'] = train_data['time'].apply(lambda x: time.localtime(x).tm_hour)
train_data['day'] = train_data['time'].apply(lambda x: time.localtime(x).tm_mday)
# 画图
Data_exploration(train_data, ['hour', 'day'], [])

Data_exploration(train_data, ["creative_type",  # 创意类型
                              "creative_is_jump",  # 是否是落地落地页跳转
                              "creative_is_download",  # 是否是落地页下载
                              "creative_has_deeplink",  # 相应素材是否有deeplink
                              "creative_is_js",  # 是否位JS素材
                              "creative_is_voicead"  # 是否为语音广告
                              ], [])

Data_exploration(train_data, [], ["creative_width", "creative_height"])

train_data[["app_cate_id", "f_channel", "app_id", "inner_slot_id", "app_paid"]].head()

Data_exploration(train_data, ["app_cate_id", "f_channel", "app_id", "inner_slot_id", "app_paid"], [])

train_data["app_id"].sample(10)

Data_exploration(train_data, ["app_id"], [], 1000)

Data_exploration(train_data, ["inner_slot_id"], [], 10000)

# 媒体广告位 字段
# 对于inner_slot_id的分析
train_data['inner_slot_id1'] = train_data['inner_slot_id'].apply(lambda x: x.split('_')[0])
# 画图
Data_exploration(train_data, ['inner_slot_id1'], [])

train_data["user_tags"].head()

"""
    Step 2: 数据预处理

"""


def Data_preprocessing(x, feat_Remv):
    """
    :param x: 数据
    :param feat_Remv: 指定需要删除的字段
    :return:
    """
    # 删除重复值
    x = x.drop_duplicates()

    # 删除指定的字段
    x = x.drop(feat_Remv, axis=1)

    # 填充缺失值
    x = x.fillna(-1)

    return x


test_data = Get_data(TEST_DATA_PATH)
# 指定需要删除的字段
feat_Remv = ["instance_id", "creative_is_js", "creative_is_voicead", "app_paid"]
train_data = Data_preprocessing(train_data, feat_Remv)

user_ids = test_data["instance_id"]
test_data = Data_preprocessing(test_data, feat_Remv)
# 训练集和测试集的划分索引
train_num, test_num = train_data.shape[0], test_data.shape[0]

# 特征和标签
X = pd.concat([train_data.drop("click", axis=1), test_data])
train_y = train_data["click"]

"""
    Step 3: 特征工程

"""


def Feature_engineering(X, y, train_num, skb_samples=10000):
    """
    :param X: 特征
    :param y: 标签
    :param train_num: 数据划分行索引节点
    :param skb_samples: 抽样样本数
    :return: 特征工程结果
    """
    assert (y is not None) or (support_index is not None)

    x = X.copy()

    print("create day and hour")
    # time 时间
    x["day"] = x["time"].apply(lambda x: time.localtime(x).tm_mday)
    x["hour"] = x["time"].apply(lambda x: time.localtime(x).tm_hour)
    del x["time"]
    print("#" * 50)

    print("create advert_industry_inner1 and advert_industry_inner2")
    # 广告主行业  advert_industry_inner
    x["advert_industry_inner_1"] = x["advert_industry_inner"].apply(lambda x: x.split("_")[0])
    x["advert_industry_inner_2"] = x["advert_industry_inner"].apply(lambda x: x.split("_")[1])
    print("#" * 50)

    print("create inner_slot_id1")
    # 媒体广告位 inner_slot_id
    x["inner_slot_id_1"] = x["inner_slot_id"].apply(lambda x: x.split("_")[0])
    del x["inner_slot_id"]
    print("#" * 50)

    print("label encoder categorical features")
    # 类别标签数值化
    categorical_features = [v for v in x.columns.values if
                            v != "creative_width" and v != "creative_height" and v != "user_tags"]
    for feature in categorical_features:
        label_encoder = preprocessing.LabelEncoder()
        x[feature] = label_encoder.fit_transform(x[feature].astype(str))
    print("#" * 50)

    print("standard scaler numerical features")
    standard_scaler = preprocessing.StandardScaler()
    numerical_features = ["creative_width", "creative_height"]
    x[numerical_features] = standard_scaler.fit(x[numerical_features]).transform(x[numerical_features])
    print("#" * 50)

    print("clear user_tags")
    x["user_tags"] = x["user_tags"].apply(lambda x: x.replace(",", " ") if x != -1 else x)
    print("#" * 50)

    print("count vectorize user_tags")
    cv = CountVectorizer()
    user_tags_sparse = cv.fit_transform(x['user_tags'].astype(str))
    x = sparse.hstack((x.drop("user_tags", axis=1), user_tags_sparse)).tocsr()
    print("#" * 50)

    # 特征选择
    print("select features")
    train_x, test_x = x[:train_num, :], x[train_num:, :]

    skb = SelectKBest(k=(100 if 100 <= x.shape[1] else x.shape[1]))
    skb.fit(train_x[:skb_samples], y[:skb_samples])
    support_index = [index for index in range(skb.get_support().shape[0]) if skb.get_support()[index]]
    train_x, test_x = train_x[:, support_index], test_x[:, support_index]
    print("#" * 50)

    return train_x, test_x


# 做特征工程(只选择1000个样本做演示)
skb_samples = 1000
train_x, test_x = Feature_engineering(X, train_y, train_num, skb_samples)

"""
Step 4: 创建模型

"""
lr = LogisticRegression()
gbdt = GradientBoostingClassifier()
xgb = XGBClassifier()
lgbm = LGBMClassifier()

"""
    Step 5: 模型参数调参

"""


def ModelOptimization(model, params, train_x, train_y):
    """
    :param model: 模型
    :param params: 参数
    :param train_x: 训练数据特征
    :param train_y: 训练数据标签
    :return: 最优模型参数

    """
    best_params = []
    for param in params:
        cv = GridSearchCV(estimator=model,  # 模型
                          param_grid=param,  # 参数列表
                          scoring="neg_log_loss",  # 评分规则
                          cv=3,  # 交叉验证次数
                          n_jobs=-1,  #
                          )
        cv.fit(train_x, train_y)
        best_params.append(cv.best_params_)
    return best_params


# cv_samples = 100000
cv_samples = 1000
# 不同模型常见参数组合
lr_params = [{"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10]},
             {"class_weight": [None, "balanced"]},
             {"penalty": ["l1", "l2"]},
             {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]

lr_best_params = ModelOptimization(lr, lr_params, train_x[:cv_samples], train_y[:cv_samples])
print("lr_best_params: ", lr_best_params)

gbdt_params = [{"n_estimators": [100, 300, 500, 1000]},  #
               {"learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
               {"max_features": [None, "log2", "sqrt"]},
               {"max_depth": [3, 5, 7, 9]},
               {"min_samples_split": [2, 4, 6, 8]},
               {"min_samples_leaf": [1, 3, 5, 7]}]
gbdt_best_params = ModelOptimization(gbdt, gbdt_params, train_x[:cv_samples], train_y[:cv_samples])
print("gbdt_best_params: ", gbdt_best_params)

xgb_params = [{"learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]},
              {"n_estimators": [100, 300, 500, 1000]},
              {"max_depth": range(3, 10, 2)},
              {"min_child_weight": range(1, 6, 2)},
              {"gamma": [i / 10.0 for i in range(0, 5)]},
              {"subsample": [i / 10.0 for i in range(6, 10)]},
              {"colsample_bytree": [i / 10.0 for i in range(6, 10)]},
              {"reg_alpha": [1e-5, 1e-2, 0.1, 1, 100]}]
xgb_best_params = ModelOptimization(xgb, xgb_params, train_x[:cv_samples], train_y[:cv_samples])
print("xgb_best_params: ", xgb_best_params)

lgbm_params = [{"learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1]},
               {"num_leaves": [10, 20, 30, 40, 50]},
               {"max_depth": [1, 3, 5, 7, 9]},
               {"n_estimators": [100, 300, 500]}]
lgbm_best_params = ModelOptimization(lgbm, lgbm_params, train_x[:cv_samples], train_y[:cv_samples])
print("lgbm_best_params: ", lgbm_best_params)

"""
    Step 6:模型评估

"""


def Model_evaluation(model, train_x, train_y):
    score = -cross_val_score(model,
                             train_x,
                             train_y,
                             scoring="neg_log_loss",
                             cv=3
                             ).mean()
    return score


# evaluation_samples = 100000
evaluation_samples = 1000

best_lr = LogisticRegression(C=3.0,
                             class_weight=None,
                             max_iter=100,
                             penalty="l2",
                             solver="newton-cg")

best_gbdt = GradientBoostingClassifier(n_estimators=300,
                                       learning_rate=0.3,
                                       max_features=None,
                                       max_depth=5,
                                       min_samples_split=8,
                                       min_samples_leaf=3)

best_xgb = XGBClassifier(learning_rate=0.3,
                         n_estimators=500,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0.3,
                         subsample=0.6,
                         colsample_bytree=0.8,
                         reg_alpha=1)

best_lgbm = LGBMClassifier(learning_rate=0.1,
                           num_leaves=20,
                           max_depth=7,
                           n_estimators=100)

lr_score = Model_evaluation(best_lr, train_x[:evaluation_samples], train_y[:evaluation_samples])
gbdt_score = Model_evaluation(best_gbdt, train_x[:evaluation_samples], train_y[:evaluation_samples])
xgb_score = Model_evaluation(best_xgb, train_x[:evaluation_samples], train_y[:evaluation_samples])
lgbm_score = Model_evaluation(best_lgbm, train_x[:evaluation_samples], train_y[:evaluation_samples])

print("lr_score: %.4f" % lr_score)
print("gbdt_score: %.4f" % gbdt_score)
print("xgb_score: %.4f" % xgb_score)
print("lgbm_score: %.4f" % lgbm_score)

"""
    Step 8: Model Ensembling

"""


def Model_ensembling(pred_y_list):
    ensembling_pred_y = np.array(pred_y_list).mean(axis=0)
    return ensembling_pred_y


# fit_samples = 100000
fit_samples = 1000
best_lr = LogisticRegression(C=3.0, class_weight=None, max_iter=100, penalty="l2", solver="newton-cg")
best_gbdt = GradientBoostingClassifier(n_estimators=300, learning_rate=0.3, max_features=None,
                                       max_depth=5, min_samples_split=8, min_samples_leaf=3)
best_xgb = XGBClassifier(learning_rate=0.3, n_estimators=500, max_depth=5, min_child_weight=1, gamma=0.3,
                         subsample=0.6, colsample_bytree=0.8, reg_alpha=1)
best_lgbm = LGBMClassifier(learning_rate=0.1, num_leaves=20, max_depth=7, n_estimators=100)

best_lr.fit(train_x[:fit_samples], train_y[:fit_samples])
lr_pred_y = best_lr.predict_proba(test_x)[:, 1]

best_gbdt.fit(train_x[:fit_samples], train_y[:fit_samples])
gbdt_pred_y = best_gbdt.predict_proba(test_x)[:, 1]

best_xgb.fit(train_x[:fit_samples], train_y[:fit_samples])
xgb_pred_y = best_xgb.predict_proba(test_x)[:, 1]

best_lgbm.fit(train_x[:fit_samples], train_y[:fit_samples])
lgbm_pred_y = best_lgbm.predict_proba(test_x)[:, 1]

pred_y_list = [lr_pred_y, gbdt_pred_y, xgb_pred_y, lgbm_pred_y]

ensembling_pred_y = Model_ensembling(pred_y_list)

ensembling_pred_y

"""
    Step 9: Submit

"""


def Submit(user_ids, pred_y):
    submission = pd.DataFrame({"instance_id": user_ids, "predicted_score": pred_y})
    submission.to_csv("../Submission/submission.csv", index=False)


# user_ids = test_data["instance_id"].values
Submit(user_ids, lr_pred_y)
Submit(user_ids, gbdt_pred_y)
Submit(user_ids, xgb_pred_y)
Submit(user_ids, lgbm_pred_y)
Submit(user_ids, ensembling_pred_y)
