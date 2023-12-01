import logging
import sys
import time
import json
import numpy as np
from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.basemodel_torch import BaseModelTorch

import numpy as np
import pandas as pd
from pathlib import Path
from models.IPS import sampling, compute_ips
from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split


def cross_validation(model, X, y, args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()
    np.random.seed(args.seed)
    X = pd.DataFrame(X)
    X["Set"] = np.random.choice(["train", "valid", "test"], p=[.8, .1, .1], size=(X.shape[0],))
    out = Path("/data/lsw/data/data/" + args.dataset + "/" + args.type + args.dataset + "_" + str(
        args.missingrate) + ".csv")
    train_index = X[X.Set != "test"].index
    test_index = X[X.Set == "test"].index

    X = X.drop(columns=['Set']).to_numpy()
    if args.model_name == "TabTransformer_our":
        datafull = pd.read_csv(
            Path("/data/lsw/data/data/" + args.dataset + "/" + "mcar_" + args.dataset + "_" + str(
                0.0) + ".csv"))
        datamiss = pd.read_csv(out)
        # 标记专家数据
        datamiss, indicator = sampling(datafull, datamiss, 20, method='feature')
        ips = compute_ips(datamiss[:, :-1], indicator[:, :-1], method='xgb')
        ips = torch.tensor(ips)
        softm = nn.Softmax(dim=1)
        ips = softm(ips)
        ips_train, ips_test = ips[train_index], ips[test_index]


    # if args.objective == "regression":
    #     kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seeda)
    # elif args.objective == "classification" or args.objective == "binary":
    #     kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    # else:
    #     raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

        # Create a new unfitted version of the model
    curr_model = model.clone()

        # Train model
    train_timer.start()
    if args.model_name == "TabTransformer_our":
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test, ips_train, ips_test)  # X_val, y_val)
    else:
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val
    train_timer.end()

        # Test model
    test_timer.start()
    # curr_model.predict(X_test, ips_test)
    if args.model_name == "TabTransformer_our":
        curr_model.predict_ips(X_test, ips_test)
    else:
        curr_model.predict(X_test)
    # curr_model.predict(X_test)
    test_timer.end()

        # Save model weights and the truth/prediction pairs for traceability
    curr_model.save_model_and_predictions(y_test)

    if save_model:
        save_loss_to_file(args, loss_history, "loss")
        save_loss_to_file(args, val_loss_history, "val_loss")

        # Compute scores on the output
    sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)
    print("dataset:", args.dataset)
    print("model:", args.model_name)
    print("type:", args.type)
    print("imp", args.imp_name)
    print("dataset:", args.dataset)
    print("seed:", args.seed)
    print("rate:", args.missingrate)
    print(sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)

    # print("Finished cross validation")
    return sc, (train_timer.get_average_time(), test_timer.get_average_time())


class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, self.args)

        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        return sc.get_objective_result()



def main_once(args):
    seeds = [0, 149669, 52983]
    datasets = ["gas"]
    missing_rate = [0.5]
    types = ["mcar_"]
    imputations = ["mean"]  # noimp表示不补全，xgboost可以不补全直接运行
    models = ["TabTransformer_our"]
    start_time = 0
    result = []
    t = time.localtime()
    t_time = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + str(t.tm_min)
    for dataset in datasets:
        for type in types:
            for imp_name in imputations:
                for model_str_name in models:
                    for rate in missing_rate:
                        temp_acc = np.array([])
                        temp_auc = np.array([])
                        temp_rmse = np.array([])
                        temp_r2 = np.array([])
                        for seed in seeds:
                            start_time = time.time()
                            args.model_name = model_str_name
                            args.dataset = dataset
                            args.type = type
                            args.imp_name = imp_name
                            args.missingrate = rate
                            if args.model_name in ["XGBoost"]:
                                args.use_gpu = False
                            # if model in ["TabTransformer", "TabTransformer_our"]:
                            #     args.imp_name = imp_name = "mean"
                            if args.dataset in ["Gesture", "drive",  "Letter"]:
                                args.objective = "classification"
                            elif args.dataset in ["temperature", "gas"]:
                                args.target_encode = False
                                args.objective = "regression"
                                args.direction = "minimize"
                            else:
                                args.objective = "binary"
                            args.seed = seed
                            X, y = load_data(args)
                            model_name = str2model(args.model_name)
                            # Run best trial again and save it!
                            parameters = args.parameters['Adult'][args.model_name]
                            model = model_name(parameters, args)
                            sc, _ = cross_validation(model, X, y, args, save_model=True)
                            print("time_cost:", time.time() - start_time)
                            temp_result = { "data": dataset, "missingrate": rate,"imp": imp_name,
                                            "model": args.model_name, "seed": seed,
                                            "type": type,
                                            "time": time.time() - start_time,
                                            "result": sc.get_results(),}
                            out = Path("/data/lsw/result/representation_allresult_" + t_time + ".json")
                            if dataset in ["HI", "News", "Gesture"]:
                                temp_acc = np.append(temp_acc, sc.accs[0])
                                temp_auc = np.append(temp_auc, sc.aucs[0])
                            elif dataset in ["temperature", "gas"]:
                                temp_rmse = np.append(temp_rmse, sc.mses[0])
                                temp_r2 = np.append(temp_r2, sc.r2s[0])
                            with open(out, "a") as f:
                                json.dump(temp_result, f, indent=4)

                                f.write("\n")
                        print("-------------------------------------------------")
                        print("dataset:", dataset)
                        print("model:", model_str_name)
                        print("type:", type)
                        print("imp", imp_name)
                        print("rate:", rate)
                        if dataset in ["HI", "News"]:
                            print("accuracy均值：", np.mean(temp_acc), "方差：", np.var(temp_acc))
                            print("auroc均值：", np.mean(temp_auc), "方差：", np.var(temp_auc))
                        elif dataset in ["temperature", "gas"]:
                            print("r2均值：", np.mean(temp_r2), "方差：", np.var(temp_r2))
                            print("rmse均值：", np.mean(temp_rmse), "方差：", np.var(temp_rmse))
                        print("-------------------------------------------------")
    # sc, time = cross_validation(model, X, y, args)


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    print(arguments)

        # Also load the best parameters
    parser = get_given_parameters_parser()
    arguments = parser.parse_args()
    main_once(arguments)