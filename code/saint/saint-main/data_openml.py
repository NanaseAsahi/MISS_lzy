import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)

def generate_new_matrix(A, percentage_to_set_ones=0.05):
    A = np.array(A)
    # Step 1: Create a zero-filled matrix B with the same shape as A
    B = np.zeros_like(A).astype(bool)

    # Step 2: Set positions with 1 in A to 0 in B
    # Step 3: Set 10% of the positions with 0 in A to 1 in B
    num_zeros = np.sum(A == False)
    num_values_to_set_one = int(num_zeros * percentage_to_set_ones)

    # Generate random indices to set to 1
    random_indices_to_set_one = np.random.choice(num_zeros, num_values_to_set_one, replace=False)
    zero_indices = np.where(A == False)
    # Set the chosen positions to 1 in B
    B[zero_indices[0][random_indices_to_set_one], zero_indices[1][random_indices_to_set_one]] = True
    return B


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def data_split(X, y, nan_mask, ips, rowips, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices],
        'ips': ips.values[indices],
        'rowips': rowips.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d

def data_prep_openml(dataset, missingrate, seed, task, datasplit=[.8, .1, .1], ips=None, rowips=None, missingtype="mcar_"):
    ds_id = 42178
    np.random.seed(seed)
    # out = Path("/home/pyc/workspace/saint-main/" + dataset + "/" + "mcar_" + dataset + "_" + str(missingrate) + ".csv")
    out = Path("/data/lsw/data/data/" + dataset + "/" + missingtype + dataset + "_" + str(missingrate) + ".csv")
    # out = Path("/data/lsw/data/" + dataset + "/" + "mnar_" + dataset + "_" + str(missingrate) + ".csv")
    # out = Path("/share/home/22251082/data/" + dataset + "/" + "mcar_" + dataset + "_" + str(missingrate) + ".csv")
    # out = Path("/data/lsw/missingdata/data/" + dataset + "/" + "mcar_" + dataset + "_" + str(missingrate) + ".csv")
    data = pd.read_csv(out)
    X = data.iloc[:, :-1]
    ips = pd.DataFrame(ips)
    rowips = pd.DataFrame(rowips)
    nunique = X.nunique()
    types = X.dtypes
    categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
    y = data.iloc[:, -1:].squeeze()
    # 分割出分类数据
    for col in X.columns:
        if types[col] == 'object' or nunique[col] < 100:
            categorical_indicator[X.columns.get_loc(col)] = True

    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))


    # 分割数据集
    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set == "train"].index
    valid_indices = X[X.Set == "valid"].index
    test_indices = X[X.Set == "test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    # nan_mask 中0为空，1为非空
    for col in categorical_columns:
        X[col] = X[col].astype("str")
    cat_dims = []
    # 填补缺失值
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)
    X_train, y_train = data_split(X, y, nan_mask, ips, rowips, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, ips, rowips, valid_indices)
    X_test, y_test = data_split(X, y, nan_mask, ips, rowips, test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std




class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X_ips = X['ips'].copy()
        rowips = X['rowips'].copy()
        rowips = rowips.squeeze()
        X = X['data'].copy()


        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns
        self.X1_ips = X_ips[:, cat_cols].copy().astype(np.float32)
        self.X2_ips = X_ips[:, con_cols].copy().astype(np.float32)
        self.rowips = rowips.copy().astype(np.float32)
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        self.cls_ips = np.ones_like(self.y, dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], np.concatenate(
            (self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx], np.concatenate(
            (self.cls_ips[idx], self.X1_ips[idx])), self.X2_ips[idx], self.rowips[idx]