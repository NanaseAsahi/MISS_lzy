import json

import pandas as pd
import numpy as np
import time
from MIWAE import MIWAE
from notmiwae import notMIWAE
from GAIN import gain_main
from missingpy import MissForest
datasets = [  "gas"]
missingrates = [0.1, 0.3, 0.5, 0.7, 0.9]
missingtypes = ["mcar_", "mnar_p_", "mar_p_"]
imp = [ "gain", "missforest"]
# missingtypes = ["mcar_"]

result_list = []
for dataset in datasets:
    for missingrate in missingrates:
        for missingtype in missingtypes:

            out = "/data/lsw/data/data/" + dataset + "/" + missingtype + dataset + "_" + str(
                missingrate) + ".csv"
            full_out = "/data/lsw/data/data/" + dataset + "/" + "mcar_" + dataset + "_" + str(
                0.0) + ".csv"
            # #
            if "miwae" in imp:
                data = pd.read_csv(out, header=0)
                full_data = pd.read_csv(full_out, header=0)
                x_full = full_data.iloc[:, :-1]
                data_x = data.iloc[:, :-1]


                time_start = time.time()
                imputed_data = MIWAE(np.array(data_x), np.array(x_full))
                imputed_data = pd.DataFrame(imputed_data)
                data.iloc[:, :-1] = imputed_data
                imp_out = "/data/lsw/data/data/" + dataset + "/" + missingtype + dataset + "_" + str(
                    missingrate) + "_miwae.csv"
                data.to_csv(imp_out, index=None)
                time_end = time.time()  # 结束计时
                time_c = time_end - time_start  # 运行所花时间
                temp_result = {"dataset": dataset, "missingrate": missingrate, "time": time_c, "method": "miwae", "type": missingtype}
                result_list.append(temp_result)
            if "notmiwae" in imp:
                data = pd.read_csv(out, header=0)
                full_data = pd.read_csv(full_out, header=0)
                x_full = full_data.iloc[:, :-1]
                data_x = data.iloc[:, :-1]

                time_start = time.time()
                S = np.array(~np.isnan(data_x), dtype=np.float32)
                imputed_data = notMIWAE(np.array(data_x), S, np.array(x_full))
                imputed_data = pd.DataFrame(imputed_data)
                data.iloc[:, :-1] = imputed_data
                imp_out =  "/data/lsw/data/data/" + dataset + "/" + missingtype + dataset + "_" + str(
                    missingrate) + "_notmiwae.csv"
                data.to_csv(imp_out, index=None)
                time_end = time.time()  # 结束计时
                time_c = time_end - time_start  # 运行所花时间
                temp_result = {"dataset": dataset, "missingrate": missingrate, "time": time_c, "method": "notmiwae", "type": missingtype}
                result_list.append(temp_result)
            if "gain" in imp:
                data = pd.read_csv(out, header=0)
                full_data = pd.read_csv(full_out, header=0)
                x_full = full_data.iloc[:, :-1]
                data_x = data.iloc[:, :-1]
                # #
                # #
                time_start = time.time()  # 开始计时
                filled_data = gain_main(data_x)
                data.iloc[:, :-1] = filled_data
                imp_out = "/data/lsw/data/data/" + dataset + "/" + missingtype + dataset + "_" + str(
                   missingrate) + "_gain.csv"
                data.to_csv(imp_out, index=None)
                time_end = time.time()  # 结束计时
                # #
                time_c = time_end - time_start  # 运行所花时间
                temp_result = {"dataset": dataset, "missingrate": missingrate, "time": time_c, "method": "gain", "type": missingtype}
                result_list.append(temp_result)
            # #
            if "missforest" in imp:
                data = pd.read_csv(out, header=0)
                full_data = pd.read_csv(full_out, header=0)
                x_full = full_data.iloc[:, :-1]
                data_x = data.iloc[:, :-1]

                time_start = time.time()  # 开始计时
                imputer = MissForest(decreasing=True, random_state=0, verbose=True, max_iter=1)
                x_filled = imputer.fit_transform(data_x)
                x_filled = pd.DataFrame(x_filled)
                x_filled["Labels"] = data.iloc[:, -1:]
                imp_out = "/data/lsw/data/data/" + dataset + "/" + missingtype + dataset + "_" + str(
                    missingrate) + "_missforest.csv"
                x_filled.to_csv(imp_out, index=None, header=data.columns)
                time_end = time.time()  # 结束计时

                time_c = time_end - time_start  # 运行所花时间
                temp_result = {"dataset": dataset, "missingrate": missingrate, "time": time_c, "method": "missforest",
                               "type": missingtype}
                result_list.append(temp_result)


            out = "/data/lsw/result/imputetime_Gas.json"
            with open(out, "w") as f:
                json.dump(result_list, f, indent=4)
