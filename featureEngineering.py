"""
    Step 3 : 特征工程
"""
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest


def Feature_engineering(X, y, train_num, skb_samples = 10000) :
    """

    :param X: 特征
    :param y: 标签
    :param train_num: 数据划分
    :param skb_samples: 抽样样本数
    :return: 特征工程结果
    """
    assert (y is not None) or (support_index is not None)

    x = X.copy()

    print("creat day and hour")
    #time 时间
    x["day"] = x["time"].apply(lambda x : time.localtime(x).tm_mday)
    x["hour"] = x["time"].apply(lambda x : time.localtime(x).tm_hour)
    del x["time"]
    print("#" * 50)


    print("create advert_industry_inner1 and advert_industry_inner2")
    #广告主行业  advert_industry_inner
    x["advert_industry_inner_1"] = x["advert_industry_inner"].apply(lambda x : x.split("_")[0])
    x["advert_industry_inner_2"] = x["advert_industry_inner"].apply(lambda x : x.split("_")[1])
    print("#" * 50)


    print("create inner_slot_id1")
    #媒体广告位 inner_slot_id
    x["inner_slot_id_1"] = x["inner_slot_id"].apply(lambda x : x.split("_")[0])
    del x["inner_slot_id"]
    print("#" * 50)


    print("label encoder categorical features")
    #类别标签数值化
    categorical_features = [v for v in x.columns.values if v != "creative_width" and v != "creative_height" and v != "user_tags"]
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
    x["user_tags"] = x["user_tags"].apply(lambda x : x.replace(",", " ") if x != -1 else x)
    print("#" * 50)

    print("count vectorize user_tags")
    cv = CountVectorizer()
    user_tags_sparse = cv.fit_transform(x['user_tags'].astype(str))
    x = sparse.hstack((x.drop("user_tags", axis = 1), user_tags_sparse)).tocsr()
    print("#" * 50)

    # 特征选择
    print("select features")
    train_x, test_x = x[:train_num, :], x[train_num:, :]

    skb = SelectKBest(k = (100 if 100 <= x.shape[1] else x.shape[1]))
    skb.fit(train_x[:skb_samples], y[:skb_samples])
    support_index = [index for index in range(skb.get_support().shape[0])if skb.get_support()[index]]
    train_x, test_x = train_x[:, support_index], test_x[:, support_index]
    print("#" * 50)

    return train_x, test_x

# 做特征工程(只选择1000个样本做演示)
skb_samples = 1000
train_x,test_x = Feature_engineering(X, train_y, train_num, skb_samples)
