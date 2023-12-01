import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd
from pathlib import Path

def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q


def load_data(args):
    print("Loading dataset " + args.dataset + "...")
    np.random.seed(args.seed)
    if args.imp_name in ["noimp"]:
        out = Path("/data/lsw/data/data/" + args.dataset + "/" + args.type + args.dataset + "_" + str(
            args.missingrate) + ".csv")
        data = pd.read_csv(out)
    elif args.imp_name in ["mean"]:
        out = Path("/data/lsw/data/data/" + args.dataset + "/" + args.type + args.dataset + "_" + str(args.missingrate) + ".csv")
        data = pd.read_csv(out)
        data = data.fillna(np.mean(data))
    else:
        out = Path("/data/lsw/data/data/" + args.dataset + "/" + args.type + args.dataset + "_" + str(
            args.missingrate) + args.imp_name + ".csv")
        data = pd.read_csv(out)
    X = data.iloc[:, :-1]
    _, args.num_features = X.shape
    nunique = X.nunique()
    types = X.dtypes
    categorical_indicator = list(np.zeros(X.shape[1]).astype(bool))
    y = data.iloc[:, -1:].squeeze()
    for col in X.columns:
        if types[col] == 'object' or nunique[col] < 100:
            categorical_indicator[X.columns.get_loc(col)] = True

    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    args.cat_idx = cat_idxs
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    X = X.to_numpy()
    y = y.to_numpy()
    # if args.dataset == "CaliforniaHousing":  # Regression dataset
    #     X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    #
    # elif args.dataset == "Covertype":  # Multi-class classification dataset
    #     X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
    #     # X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset
    #
    # elif args.dataset == "KddCup99":  # Multi-class classification dataset with categorical data
    #     X, y = sklearn.datasets.fetch_kddcup99(return_X_y=True)
    #     X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset
    #
    #     # filter out all target classes, that occur less than 1%
    #     target_counts = np.unique(y, return_counts=True)
    #     smaller1 = int(X.shape[0] * 0.01)
    #     small_idx = np.where(target_counts[1] < smaller1)
    #     small_tar = target_counts[0][small_idx]
    #     for tar in small_tar:
    #         idx = np.where(y == tar)
    #         y[idx] = b"others"
    #
    #     # new_target_counts = np.unique(y, return_counts=True)
    #     # print(new_target_counts)
    #
    #     '''
    #     # filter out all target classes, that occur less than 100
    #     target_counts = np.unique(y, return_counts=True)
    #     small_idx = np.where(target_counts[1] < 100)
    #     small_tar = target_counts[0][small_idx]
    #     for tar in small_tar:
    #         idx = np.where(y == tar)
    #         y[idx] = b"others"
    #
    #     # new_target_counts = np.unique(y, return_counts=True)
    #     # print(new_target_counts)
    #     '''
    # elif args.dataset == "Adult" or args.dataset == "AdultCat":  # Binary classification dataset with categorical data, if you pass AdultCat, the numerical columns will be discretized.
    #     url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    #
    #     features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
    #                 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    #     label = "income"
    #     columns = features + [label]
    #     df = pd.read_csv(url_data, names=columns)
    #
    #     # Fill NaN with something better?
    #     df.fillna(0, inplace=True)
    #     if args.dataset == "AdultCat":
    #         columns_to_discr = [('age', 10), ('fnlwgt', 25), ('capital-gain', 10), ('capital-loss', 10),
    #                             ('hours-per-week', 10)]
    #         for clm, nvals in columns_to_discr:
    #             df[clm] = discretize_colum(df[clm], num_values=nvals)
    #             df[clm] = df[clm].astype(int).astype(str)
    #         df['education_num'] = df['education_num'].astype(int).astype(str)
    #         args.cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    #     X = df[features].to_numpy()
    #     y = df[label].to_numpy()
    #
    # elif args.dataset == "HIGGS":  # Binary classification dataset with one categorical feature
    #     path = "/opt/notebooks/data/HIGGS.csv.gz"
    #     df = pd.read_csv(path, header=None)
    #     df.columns = ['x' + str(i) for i in range(df.shape[1])]
    #     num_col = list(df.drop(['x0', 'x21'], 1).columns)
    #     cat_col = ['x21']
    #     label_col = 'x0'
    #
    #     def fe(x):
    #         if x > 2:
    #             return 1
    #         elif x > 1:
    #             return 0
    #         else:
    #             return 2
    #
    #     df.x21 = df.x21.apply(fe)
    #
    #     # Fill NaN with something better?
    #     df.fillna(0, inplace=True)
    #
    #     X = df[num_col + cat_col].to_numpy()
    #     y = df[label_col].to_numpy()
    #
    # elif args.dataset == "Heloc":  # Binary classification dataset without categorical data
    #     path = "heloc_cleaned.csv"  # Missing values already filtered
    #     df = pd.read_csv(path)
    #     label_col = 'RiskPerformance'
    #
    #     X = df.drop(label_col, axis=1).to_numpy()
    #     y = df[label_col].to_numpy()
    #
    # else:
    #     raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")
    print(X.shape)

    # Preprocess target
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Setting this if classification task
        if args.objective == "classification":
            args.num_classes = len(le.classes_)
            print("Having", args.num_classes, "classes as target.")

    args.cat_dims = []
    con_idxs = []
    # Preprocess data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])

            # Setting this?
            args.cat_dims.append(len(le.classes_))

        else:
            con_idxs.append(i)

    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        if len(con_idxs) > 0:
            X[:, con_idxs] = scaler.fit_transform(X[:, con_idxs])

    if args.one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.cat_idx])
        if len(con_idxs) > 0:
            new_x2 = X[:, con_idxs]
            X = np.concatenate([new_x1, new_x2], axis=1)
        else:
            X = new_x1
        print("New Shape:", X.shape)

    return X, y