import random

import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


"""
MLP begins
"""
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)
        self.act2 = nn.Sigmoid()  # 二分类中 sigmoid直接输出正类概率(缺失) softmax则输出两个类别的概率列表

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout(x)

        x = self.act1(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return self.act2(x)
    
def train_mlp(model: MLP, epochs:int, dataloader:DataLoader, optimizer, loss, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:1')  # 改为cuda:1保持一致
        else:
            device = torch.device('cpu')
    
    model.to_device(device)

    model.train()

    best_loss = float('inf')
    # 早停参数 计数器大于max(10)就早停
    patience_max = 10
    count_patienece = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss_value = loss(outputs, batch_y)

            loss_value.backward()
            optimizer.step()

            epoch_loss += loss_value.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches

        # 由于表格数据容易过拟合，定义一个早停功能
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_max:
                print(f"Early stopping at epoch {epoch + 1}")
                break

def predict_mlp(model: MLP, X_predict_tensor, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:1')  # 改为cuda:1保持一致  
        else:
            device = torch.device('cpu')
    
    X_predict_tensor = X_predict_tensor.to(device)
    pred = model(X_predict_tensor)

    return pred.squeeze().cpu().numpy()  # 返回numpy数组

"""
MLP ends
"""




"""
Focal Loss begins
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        if inputs.max() > 1.0 or inputs.min() < 0.0:
            # 保证是概率
            p = torch.sigmoid(inputs)
        else:
            p = inputs
        
        # 计算二分类的 focal loss
        p_t = p * targets + (1 - p) * (1 - targets)  # p_t 是对真实类别的预测概率
        
        # 计算 alpha_t
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = self.alpha
        
        # 计算 focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # 计算 focal loss
        focal_loss = -alpha_t * focal_weight * torch.log(p_t + 1e-8)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# 当前文件用于缺失数据的人工填补，以及IPS值的计算
def sampling(datafull, datamiss, num, method='feature'):  # 选择数据进行人工填补
    datafull = np.array(datafull)
    datamiss = np.array(datamiss)

    indicator = np.array(~np.isnan(datamiss), dtype=np.float32)  # 有值为1，无值为0
    
    p = datamiss.shape[1]
    if method == 'feature':
        for i in range(p):
            if np.isnan(datamiss[:, i]).any():  # datamiss有缺失值则进行下面的步骤
                index = np.where(np.isnan(datamiss[:, i]))  # 找到缺失值的位置(找到行索引)
                slice = np.random.choice(len(index[0]), num)  # 随机选择num个样本s(N_m)
                target = index[0][slice]  # index: (array([行索引数组]),) index[0]: array([行索引])
                datamiss[target, i] = datafull[target, i]  # 用datafull的值填补datamiss
                indicator[target, i] = 0.5  # 人为填补
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

    complete_sample_index = np.random.choice(complete_data_index, size=complete_num, replace=False)  # 随机选择complete_num个完整数据的索引

    # 拼接完整数据和人为填充的数据
    train_index = np.concatenate((complete_sample_index, human_sample_index), axis=0)
    predict_index = np.setdiff1d(complete_data_index, complete_sample_index)

    return train_index, predict_index


# 论文中的实际实现方式
def train_test_split_no_random(datamiss, indicator, column, sample_num=20, complete_num=20):
    # 本函数用于完整数据的采样
    human_sample_index = np.where(indicator[:, column] == 0.5)[0]  # 填补的数据的索引
    imputation_data = pd.DataFrame(datamiss[human_sample_index, column])  # 填补的数据
    imputation_mean = imputation_data.mean()  # 计算填补数据的均值

    # 以0.5为对称轴，找到对称的点
    complete_mean = 1 - imputation_mean  # 完整数据的均值

    complete_data_index = np.where(indicator[:, column] == 1)[0]  # 未缺失数据的索引
    complete_data = pd.DataFrame(datamiss[complete_data_index, column], index=complete_data_index)  # 完整的数据
    # 计算完整数据与均值的绝对差值
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
        if (indicator[:, i] == 0.5).any():  # 第i列经过人工填补
            index = np.where(indicator[:, i] != 0)  # 第i列未缺失或者经过人工填补后的行号 / i.e. =.5 or 1

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
                # xgb不需要处理缺失值 避免了插补带来的信息损失
                # "scale_pos_weight"用于处理类别不平衡问题 未缺失则概率为0，若缺失则概率为1
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
            elif method == 'mlp':
                # 填补缺失值
                data_X_train_filled = data_X_train.copy()
                data_X_predict_filled = data_X_predict.copy()

                for col in data_X_train_filled.columns:
                    if data_X_train_filled[col].isnull().any():
                        data_X_train_filled[col].fillna(value=data_X_train_filled[col].mean(), inplace=True)
                
                for col in data_X_predict_filled.columns:
                    if data_X_predict_filled[col].isnull().any():
                        data_X_predict_filled[col].fillna(value=data_X_train_filled[col].mean(), inplace=True)

                # 标准化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(data_X_train_filled.values)
                X_predict_scaled = scaler.transform(data_X_predict_filled.values)
                
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.FloatTensor(y_train)
                X_predict_tensor = torch.FloatTensor(X_predict_scaled)

                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                in_dim = X_train_tensor.shape[1]
                model = MLP(in_dim, 1)

                # 可替换为FocalLoss等
                loss = nn.BCELoss()
                optimizer = Adam(model.parameters(), lr=0.001)

                train_mlp(model, epochs=20, dataloader=train_loader, optimizer=optimizer, loss=loss)

                model.eval()

                pred = predict_mlp(model, X_predict_tensor)
                p_miss[predict_index, i] = pred
                p_miss[train_index, i] = y_train
                
    # p_miss太接近1会导致ips太大
    p_miss[(p_miss) > 0.95] = 0.95  # 缺失的概率最高值设置为0.95
    ips = 1 / (1 - p_miss)  # 计算逆概率
    ips[(indicator == 0)] = 0  # indicator == 0表示数据存在缺失，如果数据存在缺失则将其IPS设置为0

    return ips