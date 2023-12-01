import pandas as pd
import numpy as np
import math, random
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# headers = ["Wine", "Alcohol", "Malic.acid", "Ash", "Acl", "Mg", "Phenols", "Flavanoids", "Nonflavanoid.phenols",
#            "Proanth", "Color.int", "Hue", "OD", "Proline"]


# mcar
def missing_sim_mcar(dataset_name="Wine", seed=None, headers=None, missing_rates=[0.5]):
    out = Path("/share/home/22251082/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_0.0" ".csv")
    o_data = pd.read_csv(out)
    if headers is not None:
        o_data.columns = headers
    else:
        headers = o_data.columns
    row, col = o_data.shape
    for missing_rate in missing_rates:
        data = o_data.copy()
        for col in headers:
            print(col)
            if col == "Labels" or col == "shares" or col == "target":
                continue
            arr = np.arange(row)
            if seed:
                np.random.seed(seed)
            np.random.shuffle(arr)
            na_size = math.ceil(row * missing_rate)
            na = np.arange(start=0, stop=na_size)
            arr = arr[na]
            data.loc[arr, col] = np.NAN

        mask = ~data.isnull() * 1
        # data = data.sample(frac=1.0)
        data_name = "mcar_" + dataset_name + "_" + str(missing_rate) + ".csv"
        # data_train.to_csv("data/train_" + data_name, index=False)
        # data_test.to_csv("data/test_" + data_name, index=False)
        data.to_csv("/share/home/22251082/data/" + dataset_name + "/" + data_name , index=None)


# mar
def missing_sim_mar(dataset_name="Wine.csv", seed=None, headers=None, missing_rates=[0.5]):
    out = Path("/share/home/22251082/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_0.0" ".csv")
    o_data = pd.read_csv(out)
    correlation_matrix = o_data.iloc[:, :-1].corr()
    # 查找与每一列最相关的列
    correlation_matrix = correlation_matrix.abs()
    for i in range(correlation_matrix.shape[1]):
        correlation_matrix.iloc[i,i] = 0
    most_correlated_col = correlation_matrix.idxmax()

    # 打印每一列与其最相关的列
    print(most_correlated_col)
    if headers is not None:
        o_data.columns = headers
    else:
        headers = o_data.columns
    row, col = o_data.shape
    for missing_rate in missing_rates:
        data = o_data.copy()
        mar_data = o_data.copy()
        for col in range(len(headers)):
            if headers[col] == "Labels" or headers[col] == "shares" or headers[col] == "target":
                continue
            data["oldIndex"] = data.index
            # print(col, headers[col], headers[(col + 1) % (len(headers) - 1)])
            data = data.sort_values(by=most_correlated_col[headers[col]], ascending=False)
            data.index = range(0, row)
            # median_index = math.ceil(row * 1.0 / 2)
            # # randomly select the part larger or smaller than the median
            # part_rand = 0
            # # 1: smaller part
            # left_bound = 0 if part_rand else median_index
            # right_bound = median_index - 1 if part_rand else row - 1
            na_size = math.ceil(row * missing_rate)
            arr = np.arange(start=0, stop=na_size + 1)
            # delete 2*missing_rate*size according to random index of half of the column
            # for i in range(0, na_size + 1):
            data.loc[arr, headers[col]] = np.NAN
            # restore to default order
            data = data.sort_values(by=["oldIndex"])
            data.index = range(row)
            mar_data.loc[:, headers[col]] = data.loc[:, headers[col]]
            data = o_data.copy()

        # assert data == 0
        data_name = "mar_" + dataset_name + "_" + str(missing_rate) + ".csv"
        marout = Path("/share/home/22251082/data/" + dataset_name + "/" + data_name)
        mar_data.to_csv(marout, index=None)


# mnar
def missing_sim_mnar(dataset_name="Wine.csv", seed=None, headers=None, missing_rates=[0.5]):
    out = Path("/data/lsw/data/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_0.0" ".csv")
    o_data = pd.read_csv(out)
    if headers is not None:
        o_data.columns = headers
    else:
        headers = o_data.columns
    row, col = o_data.shape
    for missing_rate in missing_rates:
        data = o_data.copy()

        for col in headers:
            if col == "Labels" or col == "shares" or col == "target":
                continue
            data["oldIndex"] = data.index
            data = data.sort_values(by=col, ascending=True)
            data.index = range(0, row)
            # median_index = math.ceil(row * 1.0 / 2)
            # # randomly select the part larger or smaller than the median
            # part_rand = 0
            # # 1: smaller part
            # left_bound = 0 if part_rand else median_index
            # right_bound = median_index - 1 if part_rand else row - 1
            na_size = math.ceil(row * missing_rate)
            arr = np.arange(start=0, stop=na_size + 1)
            # delete 2*missing_rate*size according to random index of half of the column
            data.loc[arr, col] = np.NAN
            # restore to default order
            data = data.sort_values(by=["oldIndex"])
            data.index = range(row)
            data = data.iloc[:, :-1]
        # mask = ~data.isnull() * 1
        # # print(mask)
        # mask_name = "mnar_wine_mask_" + str(missing_rate) + ".csv"
        # mask.to_csv(mask_name, index=False)
        # data_name = "mnar_wine_data_" + str(missing_rate) + ".csv"
        # data.to_csv(data_name, index=None)
        data_name = "mnar_" + dataset_name + "_" + str(missing_rate) + "_reverse.csv"
        # data_train.to_csv("data/train_" + data_name, index=False)
        # data_test.to_csv("data/test_" + data_name, index=False)
        data.to_csv("/data/lsw/data/data/" + dataset_name + "/" + data_name , index=None)

def missing_sim_mnar_probability(dataset_name="Wine.csv", seed=None, headers=None, missing_rates=[0.5]):
    out = Path("/data/lsw/data/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_0.0" ".csv")
    o_data = pd.read_csv(out)
    if headers is not None:
        o_data.columns = headers
    else:
        headers = o_data.columns
    row, col = o_data.shape
    sum_row = sum(range(1, row))
    for missing_rate in missing_rates:
        data = o_data.copy()

        for col in headers:
            if col == "Labels" or col == "shares" or col == "target":
                continue
            data["oldIndex"] = data.index
            data = data.sort_values(by=col, ascending=True)
            data.index = range(0, row)
            probability = data.index * 1.0 / sum_row

            # median_index = math.ceil(row * 1.0 / 2)
            # # randomly select the part larger or smaller than the median
            # part_rand = 0
            # # 1: smaller part
            # left_bound = 0 if part_rand else median_index
            # right_bound = median_index - 1 if part_rand else row - 1
            na_size = math.ceil(row * missing_rate)
            selected_values = np.random.choice(row, na_size, replace=False, p=probability)
            # delete 2*missing_rate*size according to random index of half of the column
            data.loc[selected_values, col] = np.NAN
            # restore to default order
            data = data.sort_values(by=["oldIndex"])
            data.index = range(row)
            data = data.iloc[:, :-1]
        # mask = ~data.isnull() * 1
        # # print(mask)
        # mask_name = "mnar_wine_mask_" + str(missing_rate) + ".csv"
        # mask.to_csv(mask_name, index=False)
        # data_name = "mnar_wine_data_" + str(missing_rate) + ".csv"
        # data.to_csv(data_name, index=None)
        data_name = "mnar_p_" + dataset_name + "_" + str(missing_rate) + ".csv"
        # data_train.to_csv("data/train_" + data_name, index=False)
        # data_test.to_csv("data/test_" + data_name, index=False)
        data.to_csv("/data/lsw/data/data/" + dataset_name + "/" + data_name , index=None)

def label_missing(dataset_name="Wine.csv", missing_rates=[0.5]):

    for missing_rate in missing_rates:
        np.random.seed(0)
        out = Path("/data/lsw/data/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_" + str(missing_rate) +".csv")
        o_data = pd.read_csv(out)
        o_data["Set"] = np.random.choice(["train", "test"], p=[.9, .1], size=(o_data.shape[0],))
        train_indices = o_data[o_data.Set == "train"].index
        test_indices = o_data[o_data.Set == "test"].index
        with open("/data/lsw/data/data/" + dataset_name + "/Train_Indices_" + str(missing_rate) + ".txt",
                  'w') as f:
            f.write('\n'.join(map(str, train_indices)))

        # Save Test_Indices to a new txt file
        with open("/data/lsw/data/data/" + dataset_name + "/Test_Indices_" + str(missing_rate) + ".txt",
                  'w') as f:
            f.write('\n'.join(map(str, test_indices)))
        headers = o_data.columns
        row, col = o_data.shape
        data = o_data.copy()

        for col in headers:
            print(col)
            if col == "Labels" or col == "shares" or col == "target":
                arr = np.array(train_indices)  # Only modify rows in the train set
                np.random.shuffle(arr)
                na_size = math.ceil(len(train_indices) * missing_rate)
                na = arr[:na_size]
                data.loc[na, col] = np.NaN


        mask = ~data.isnull() * 1
        # data = data.sample(frac=1.0)
        data = data.drop(columns=['Set'])
        data_name = "mcar_label_" + dataset_name + "_" + str(missing_rate) + ".csv"
        # data_train.to_csv("data/train_" + data_name, index=False)
        # data_test.to_csv("data/test_" + data_name, index=False)
        data.to_csv("/data/lsw/data/data/" + dataset_name + "/" + data_name, index=None)
def missing_sim_mnar_probability_reverse(dataset_name="Wine.csv", seed=None, headers=None, missing_rates=[0.5]):
    out = Path("/data/lsw/data/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_0.0" ".csv")
    o_data = pd.read_csv(out)
    if headers is not None:
        o_data.columns = headers
    else:
        headers = o_data.columns
    row, col = o_data.shape
    sum_row = sum(range(1, row))
    for missing_rate in missing_rates:
        data = o_data.copy()

        for col in headers:
            if col == "Labels" or col == "shares" or col == "target":
                continue
            data["oldIndex"] = data.index
            data = data.sort_values(by=col, ascending=False)
            data.index = range(0, row)
            probability = data.index * 1.0 / sum_row

            # median_index = math.ceil(row * 1.0 / 2)
            # # randomly select the part larger or smaller than the median
            # part_rand = 0
            # # 1: smaller part
            # left_bound = 0 if part_rand else median_index
            # right_bound = median_index - 1 if part_rand else row - 1
            na_size = math.ceil(row * missing_rate)
            selected_values = np.random.choice(row, na_size, replace=False, p=probability)
            # delete 2*missing_rate*size according to random index of half of the column
            data.loc[selected_values, col] = np.NAN
            # restore to default order
            data = data.sort_values(by=["oldIndex"])
            data.index = range(row)
            data = data.iloc[:, :-1]
        # mask = ~data.isnull() * 1
        # # print(mask)
        # mask_name = "mnar_wine_mask_" + str(missing_rate) + ".csv"
        # mask.to_csv(mask_name, index=False)
        # data_name = "mnar_wine_data_" + str(missing_rate) + ".csv"
        # data.to_csv(data_name, index=None)
        data_name = "mnar_p_reverse_" + dataset_name + "_" + str(missing_rate) + ".csv"
        # data_train.to_csv("data/train_" + data_name, index=False)
        # data_test.to_csv("data/test_" + data_name, index=False)
        data.to_csv("/data/lsw/data/data/" + dataset_name + "/" + data_name , index=None)

def missing_sim_mar_probability(dataset_name="Wine.csv", seed=None, headers=None, missing_rates=[0.5]):
    out = Path("/data/lsw/data/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_0.0" ".csv")
    o_data = pd.read_csv(out)
    correlation_matrix = o_data.iloc[:, :-1].corr()
    # 查找与每一列最相关的列
    correlation_matrix = correlation_matrix.abs()
    for i in range(correlation_matrix.shape[1]):
        correlation_matrix.iloc[i,i] = 0
    most_correlated_col = correlation_matrix.idxmax()

    # 打印每一列与其最相关的列
    print(most_correlated_col)
    if headers is not None:
        o_data.columns = headers
    else:
        headers = o_data.columns
    row, col = o_data.shape
    sum_row = sum(range(1, row))
    for missing_rate in missing_rates:
        data = o_data.copy()
        mar_data = o_data.copy()
        for col in range(len(headers)):
            if headers[col] == "Labels" or headers[col] == "shares" or headers[col] == "target":
                continue
            data["oldIndex"] = data.index
            # print(col, headers[col], headers[(col + 1) % (len(headers) - 1)])
            data = data.sort_values(by=most_correlated_col[headers[col]], ascending=True)
            data.index = range(0, row)
            probability = data.index * 1.0 / sum_row
            # median_index = math.ceil(row * 1.0 / 2)
            # # randomly select the part larger or smaller than the median
            # part_rand = 0
            # # 1: smaller part
            # left_bound = 0 if part_rand else median_index
            # right_bound = median_index - 1 if part_rand else row - 1
            na_size = math.ceil(row * missing_rate)
            selected_values = np.random.choice(row, na_size, replace=False, p=probability)
            # delete 2*missing_rate*size according to random index of half of the column
            # for i in range(0, na_size + 1):
            data.loc[selected_values, headers[col]] = np.NAN
            # restore to default order
            data = data.sort_values(by=["oldIndex"])
            data.index = range(row)
            mar_data.loc[:, headers[col]] = data.loc[:, headers[col]]
            data = o_data.copy()

        # assert data == 0
        data_name = "mar_p_" + dataset_name + "_" + str(missing_rate) + ".csv"
        mar_data.to_csv("/data/lsw/data/data/" + dataset_name + "/" + data_name , index=None)


def missing_sim_mar_probability_reverse(dataset_name="Wine.csv", seed=None, headers=None, missing_rates=[0.5]):
    out = Path("/data/lsw/data/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_0.0" ".csv")
    o_data = pd.read_csv(out)
    correlation_matrix = o_data.iloc[:, :-1].corr()
    # 查找与每一列最相关的列
    correlation_matrix = correlation_matrix.abs()
    for i in range(correlation_matrix.shape[1]):
        correlation_matrix.iloc[i,i] = 0
    most_correlated_col = correlation_matrix.idxmax()

    # 打印每一列与其最相关的列
    print(most_correlated_col)
    if headers is not None:
        o_data.columns = headers
    else:
        headers = o_data.columns
    row, col = o_data.shape
    sum_row = sum(range(1, row))
    for missing_rate in missing_rates:
        data = o_data.copy()
        mar_data = o_data.copy()
        for col in range(len(headers)):
            if headers[col] == "Labels" or headers[col] == "shares" or headers[col] == "target":
                continue
            data["oldIndex"] = data.index
            # print(col, headers[col], headers[(col + 1) % (len(headers) - 1)])
            data = data.sort_values(by=most_correlated_col[headers[col]], ascending=False)
            data.index = range(0, row)
            probability = data.index * 1.0 / sum_row
            # median_index = math.ceil(row * 1.0 / 2)
            # # randomly select the part larger or smaller than the median
            # part_rand = 0
            # # 1: smaller part
            # left_bound = 0 if part_rand else median_index
            # right_bound = median_index - 1 if part_rand else row - 1
            na_size = math.ceil(row * missing_rate)
            selected_values = np.random.choice(row, na_size, replace=False, p=probability)
            # delete 2*missing_rate*size according to random index of half of the column
            # for i in range(0, na_size + 1):
            data.loc[selected_values, headers[col]] = np.NAN
            # restore to default order
            data = data.sort_values(by=["oldIndex"])
            data.index = range(row)
            mar_data.loc[:, headers[col]] = data.loc[:, headers[col]]
            data = o_data.copy()

        # assert data == 0
        data_name = "mar_p_reverse_" + dataset_name + "_" + str(missing_rate) + ".csv"
        mar_data.to_csv("/data/lsw/data/data/" + dataset_name + "/" + data_name , index=None)


# missing_sim_mnar("News")
missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
datasets = ["News", "temperature"]
for dataset in datasets:
    label_missing(dataset, missing_rates)