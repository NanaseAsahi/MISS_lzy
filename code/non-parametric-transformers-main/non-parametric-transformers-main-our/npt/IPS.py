import random

import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB


# 当前文件用于缺失数据的人工填补，以及IPS值的计算
def sampling(datafull, datamiss, num, method='feature'):  # 选择数据进行人工填补
    datafull = np.array(datafull)
    datamiss = np.array(datamiss)
    indicator = np.array(~np.isnan(datamiss), dtype=np.float32)
    p = datamiss.shape[1]
    if method == 'feature':
        for i in range(p):
            if np.isnan(datamiss[:, i]).any():
                index = np.where(np.isnan(datamiss[:, i]))
                slice = np.random.choice(len(index[0]), num)
                target = index[0][slice]
                datamiss[target, i] = datafull[target, i]
                indicator[target, i] = 0.5
    elif method == 'sample':
        index = np.where(np.isnan(datamiss).any(axis=1))
        slice = np.random.choice(len(index[0]), num)
        target = index[0][slice]
        for i in target:
            for j in range(p):
                if indicator[i, j] == 0:
                    indicator[i, j] = 0.5
    return datamiss, indicator


def train_test_split_random(datamiss, indicator, column, sample_num=20, complete_num=20):
    # 随机采样complete data
    index = np.where(indicator[:, column] != 0)  # 第column列未缺失或者经过人工填补后的行号
    complete_data_index = np.where(indicator[:, column] == 1)[0]  # column列未缺失的数据
    human_sample_index = np.where(indicator[:, column] == 0.5)[0]  # column列填补的数据

    complete_sample_index = np.random.choice(complete_data_index, size=complete_num, replace=False)
    train_index = np.concatenate((complete_sample_index, human_sample_index), axis=0)
    predict_index = np.setdiff1d(complete_data_index, complete_sample_index)

    return train_index, predict_index


def train_test_split_no_random(datamiss, indicator, column, sample_num=20, complete_num=20):
    # 本函数用于完整数据的采样
    human_sample_index = np.where(indicator[:, column] == 0.5)[0]  # 填补的数据的索引
    imputation_data = pd.DataFrame(datamiss[human_sample_index, column])  # 填补的数据
    imputation_mean = imputation_data.mean()  # 计算填补数据的均值

    # 以0.5为对称轴，找到对称的点
    complete_mean = 1 - imputation_mean  # 完整数据的均值

    complete_data_index = np.where(indicator[:, column] == 1)[0]  # 未缺失数据的索引
    complete_data = pd.DataFrame(datamiss[complete_data_index, column], index=complete_data_index)  # 完整的数据
    df_abs_diff = complete_data.sub(complete_mean).abs().sort_values(by=0, ascending=False)

    # 将绝对差值按升序排序，并获取最接近A的n个点的索引
    closest_indices = np.array(df_abs_diff[:complete_num].index)

    train_index = np.concatenate((closest_indices, human_sample_index), axis=0)
    predict_index = np.setdiff1d(complete_data_index, closest_indices)

    return train_index, predict_index


def compute_ips(datamiss, indicator, num=20, method='lr', observed_num=1, complete_sample="no-Random"):  # 计算IPS值
    n = datamiss.shape[0]
    p = datamiss.shape[1]
    p_miss = np.zeros((n, p))  # 最终的结果
    for i in range(p):
        if (indicator[:, i] == 0.5).any():  # 第i列存在缺失数据
            index = np.where(indicator[:, i] != 0)  # 第i列未缺失或者经过人工填补后的行号
            data_x = pd.DataFrame(datamiss[index, :][0])  # X
            if complete_sample == "Random":
                train_index, predict_index = train_test_split_random(datamiss, indicator, i, sample_num=num,
                                                                     complete_num=num * observed_num)
            else:
                train_index, predict_index = train_test_split_no_random(datamiss, indicator, i, sample_num=num,
                                                                        complete_num=num * observed_num)

            data_X_train = pd.DataFrame(datamiss[train_index, :])
            data_X_predict = pd.DataFrame(datamiss[predict_index, :])
            y = np.array((indicator[:, i] != 1), dtype=np.float32)  # 未缺失则概率为0，若缺失则概率为1

            y_train = y[train_index]
            y_predict = y[predict_index]

            if method == 'xgb':
                params = {'objective': 'reg:logistic', 'booster': 'gbtree', 'max_depth': 3, 'verbosity': 1,
                          'scale_pos_weight': sum(y == 0) / sum(y == 1)}
                xgb_model = xgb.train(params, xgb.DMatrix(data_X_train, y_train))
                # print(xgb_model.predict(xgb.DMatrix(data_X_train)))
                train_pred = xgb_model.predict(xgb.DMatrix(data_X_predict))
                p_miss[predict_index, i] = xgb_model.predict(xgb.DMatrix(data_X_predict))
                p_miss[train_index, i] = y_train

            elif method == 'lr':

                data_X_train = pd.DataFrame(datamiss[train_index, :])
                data_X_predict = pd.DataFrame(datamiss[predict_index, :])

                for col in data_X_train.columns:  # 遍历每一列
                    if data_X_train[col].isnull().any(axis=0):  # 其他列存在缺失数据则使用均值进行填补
                        data_X_train[col].fillna(value=int(data_X_train[col].mean()), inplace=True)
                    if data_X_predict[col].isnull().any(axis=0):  # 其他列存在缺失数据则使用均值进行填补
                        data_X_predict[col].fillna(value=int(data_X_predict[col].mean()), inplace=True)

                LR_model = LR(class_weight='balanced')
                # LR_model.fit(data_x, y)
                LR_model.fit(data_X_train, y_train)
                p_miss[predict_index, i] = LR_model.predict_proba(data_X_predict)[:, 1]
                p_miss[train_index, i] = y_train
                print(f"LR_model.coef_:{LR_model.coef_}")
                print(f"LR_model.intercept_:{LR_model.intercept_}")
            elif method == 'bayes':
                for col in data_x.columns:
                    if data_x[col].isnull().any(axis=0):
                        data_x[col].fillna(value=int(data_x[col].mean()), inplace=True)  # 其他列存在缺失则使用均值进行填补

                bayes_model = GaussianNB()
                bayes_model.fit(data_x, y)
                p_miss[index, i] = bayes_model.predict_proba(data_x)[:, 1]

    p_miss[(p_miss) > 0.95] = 0.95  # 缺失的概率最高为0.95
    ips = 1 / (1 - p_miss)  # 计算逆概率
    ips[(indicator == 0)] = 0  # indicator == 0表示数据存在缺失，如果数据存在缺失则将其IPS设置为0

    return ips