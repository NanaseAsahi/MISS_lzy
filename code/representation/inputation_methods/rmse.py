import numpy as np
import pandas as pd


datasets = ["News", "HI", "Gesture"]
missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
types = ["mcar_", "mnar_", "mar_"]
imp = ["miwae", "notmiwae", "gain", "missforest"]
for dataset in datasets:
    actual_data = "/data/lsw/data/data/" + dataset + "/" + "mcar_" + dataset + "_" + str(
        0.0) + ".csv"
    for missingtype in types:
        for imp_name in imp:
            for missingrate in missing_rates:
                imputed_data = "/data/lsw/data/data/" + dataset + "/" + missingtype + dataset + "_" + str(
                    missingrate) + ".csv"
                imputed_data = pd.read_csv(imputed_data)
                actual_data = pd.read_csv(actual_data)
                # 提取补全值和真实值的数据
                imputed_values = imputed_data['Value'].values
                actual_values = actual_data['Value'].values

                # 计算差值（残差）
                residuals = actual_values - imputed_values

                # 计算平方差
                squared_residuals = residuals ** 2

                # 计算平均平方差
                mse = np.mean(squared_residuals)

                # 计算RMSE
                rmse = np.sqrt(mse)

                print('dataset', dataset, 'missingtype', missingtype, 'imp', imp_name, 'rate', missingrate, 'RMSE:', rmse)

