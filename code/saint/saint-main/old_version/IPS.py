import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB

# 当前文件用于缺失数据的人工填补，以及IPS值的计算
def sampling(datafull, datamiss, num, method='feature'): # 选择数据进行人工填补
    datafull = np.array(datafull)
    datamiss = np.array(datamiss)
    indicator = np.array(~np.isnan(datamiss), dtype=np.float32)
    p = datamiss.shape[1]
    if method == 'feature':
        for i in range(p):
            if np.isnan(datamiss[:,i]).any():
                index = np.where(np.isnan(datamiss[:,i]))
                slice = np.random.choice(len(index[0]),num)
                target = index[0][slice]
                datamiss[target,i] = datafull[target,i]
                indicator[target,i] = 0.5
    elif method == 'sample':
        index = np.where(np.isnan(datamiss).any(axis=1))
        slice = np.random.choice(len(index[0]),num)
        target = index[0][slice]
        for i in target:
            for j in range(p):
                if indicator[i,j] == 0:
                    indicator[i,j] = 0.5
    return datamiss, indicator

def compute_ips(datamiss, indicator, method='lr'): # 计算IPS值
    # col = datamiss.columns
    n = datamiss.shape[0]
    p = datamiss.shape[1]
    p_miss = np.zeros((n, p)) # 最终的结果
    for i in range(p):
        if (indicator[:, i] == 0.5).any(): # 缺失的行数
            index = np.where(indicator[:, i] != 0) # 第i列未缺失或者经过人工填补后的行号
            data_x = pd.DataFrame(datamiss[index, :][0])  # X
            y = np.array((indicator[:, i] != 1), dtype=np.float32)
            y = y[index] # y值

            if method == 'xgb':
                params = {'objective': 'reg:logistic', 'booster': 'gbtree', 'max_depth': 3, 'silent': 1,
                          'scale_pos_weight': sum(y == 0) / sum(y == 1)}
                xgb_model = xgb.train(params, xgb.DMatrix(data_x, y))
                train_pred = xgb_model.predict(xgb.DMatrix(data_x))
                p_miss[index, i] = xgb_model.predict(xgb.DMatrix(data_x))

            elif method == 'lr':
                for col in data_x.columns: # 遍历每一列
                    if data_x[col].isnull().any(axis=0): # 其他列存在缺失数据则使用均值进行填补
                        data_x[col].fillna(value=int(data_x[col].mean()), inplace=True)

                LR_model = LR(class_weight='balanced')
                LR_model.fit(data_x, y)
                p_miss[index, i] = LR_model.predict_proba(data_x)[:, 1]
            elif method == 'bayes':
                for col in data_x.columns:
                    if data_x[col].isnull().any(axis=0):
                        data_x[col].fillna(value=int(data_x[col].mean()), inplace=True) # 其他列存在缺失则使用均值进行填补

                bayes_model = GaussianNB()
                bayes_model.fit(data_x, y)
                p_miss[index, i] = bayes_model.predict_proba(data_x)[:, 1]

    p_miss[(p_miss) > 0.95] = 0.95 # 缺失的概率最高为0.95
    ips = 1 / (1 - p_miss) # 计算逆概率
    ips[(indicator == 0)] = 0
    print(ips, ips.shape)
    return ips