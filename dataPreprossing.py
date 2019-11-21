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
    x = x.drop(feat_Remv, axis = 1)

    # 填充缺失值
    x = x.fillna(-1)

    return x


test_data = Get_data(TEST_DATA_PATH)
#指定需要删除的字段
feat_Remv = ["instance_id", "creative_is_js", "creative_is_voicead", "app_paid"]
train_data = Data_preprocessing(train_data,feat_Remv)

user_ids=test_data["instance_id"]
test_data =  Data_preprocessing(test_data,feat_Remv)
#训练集和测试集的划分索引
train_num, test_num = train_data.shape[0], test_data.shape[0]

#特征和标签
X = pd.concat([train_data.drop("click", axis = 1), test_data])
train_y = train_data["click"]
