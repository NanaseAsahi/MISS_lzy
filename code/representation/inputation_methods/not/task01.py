"""
Use the MIWAE and not-MIWAE on UCI data
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from notMIWAE import notMIWAE
import trainer
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def introduce_mising(X):
    N, D = X.shape
    Xnan = X.copy()

    # ---- MNAR in D/2 dimensions
    mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0

    return Xnan, Xz


# ---- data settings
name = '/tmp/uci/task01/best'
n_hidden = 128
n_samples = 20
max_iter = 10000
batch_size = 128
L = 10000

# ---- choose the missing model
# mprocess = 'linear'
# mprocess = 'selfmasking'
mprocess = 'selfmasking_known'

# ---- number of runs
runs = 1
RMSE_miwae = []
RMSE_notmiwae = []
RMSE_mean = []
RMSE_mice = []
RMSE_RF = []

for _ in range(runs):

    # ---- load data
    # white wine
    dataset = "HI"
    missingrate = 0.5
    out = "/data/lsw/representations/data/" + dataset + "/" + "mcar_" + dataset + "_" + str(missingrate) + ".csv"
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    data = np.array(pd.read_csv(out))
    # ---- drop the classification attribute
    data = data[:, :-1]
    # ----

    N, D = data.shape

    dl = D - 1

    # ---- standardize data
    datamean = np.nanmean(data, axis=0)
    datastd = np.nanstd(data, axis=0)
    data = data - datamean
    data = data / datastd

    # ---- random permutation
    # p = np.random.permutation(N)
    # data = data[p, :]

    Xtrain = data.copy()
    Xval_org = data.copy()

    # ---- introduce missing process
    Xnan = Xtrain
    Xz = Xtrain
    Xz[np.isnan(Xnan)] = 0
    S = np.array(~np.isnan(Xnan), dtype=np.float)
    # Xval, Xvalz = introduce_mising(Xval_org)
    Xval = Xtrain
    Xvalz = Xtrain
    Xvalz[np.isnan(Xnan)] = 0

    # ---------------------- #
    # ---- fit not-MIWAE---- #
    # ---------------------- #
    notmiwae = notMIWAE(Xnan, Xval, n_latent=dl, n_samples=n_samples, n_hidden=n_hidden, missing_process=mprocess, name=name)

    # ---- do the training
    trainer.train(notmiwae, batch_size=batch_size, max_iter=max_iter, name=name + 'notmiwae')

    # ---- find imputation RMSE
    print(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[1])
    RMSE_notmiwae.append(utils.not_imputationRMSE(notmiwae, Xtrain, Xz, Xnan, S, L)[0])

    # # ------------------------- #
    # # ---- mean imputation ---- #
    # # ------------------------- #
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp.fit(Xnan)
    # Xrec = imp.transform(Xnan)
    # RMSE_mean.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))
    #
    # # ------------------------- #
    # # ---- mice imputation ---- #
    # # ------------------------- #
    # imp = IterativeImputer(max_iter=10, random_state=0)
    # imp.fit(Xnan)
    # Xrec = imp.transform(Xnan)
    # RMSE_mice.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))
    #
    # # ------------------------------- #
    # # ---- missForest imputation ---- #
    # # ------------------------------- #
    # estimator = RandomForestRegressor(n_estimators=100)
    # imp = IterativeImputer(estimator=estimator)
    # imp.fit(Xnan)
    # Xrec = imp.transform(Xnan)
    # RMSE_RF.append(np.sqrt(np.sum((Xtrain - Xrec) ** 2 * (1 - S)) / np.sum(1 - S)))

    print('RMSE, MIWAE {0:.5f}, notMIWAE {1:.5f}, MEAN {2:.5f}, MICE {3:.5f}, missForest {4:.5f}'
          .format(RMSE_miwae[-1], RMSE_notmiwae[-1], RMSE_mean[-1], RMSE_mice[-1], RMSE_RF[-1]))


print("RMSE_miwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_miwae), np.std(RMSE_miwae)))
print("RMSE_notmiwae = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_notmiwae), np.std(RMSE_notmiwae)))
print("RMSE_mean = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_mean), np.std(RMSE_mean)))
print("RMSE_mice = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_mice), np.std(RMSE_mice)))
print("RMSE_missForest = {0:.5f} +- {1:.5f}".format(np.mean(RMSE_RF), np.std(RMSE_RF)))