import torch
import torchvision
import torch.nn as nn
import numpy as np
import scipy.stats
import scipy.io
import scipy.sparse
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributions as td
import time
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def MIWAE(data, odata):
    def mse(xhat,xtrue,mask): # MSE function for imputations
        xhat = np.array(xhat)
        xtrue = np.array(xtrue)
        return np.mean(np.power(xhat-xtrue,2)[~mask])

    # from sklearn.datasets import load_breast_cancer
    # data = load_breast_cancer(True)[0]
    # print(data)
    # xfull = (data - np.mean(data,0))/np.std(data,0)
    # n = xfull.shape[0] # number of observations
    # p = xfull.shape[1] # number of features

    # np.random.seed(1234)

    # perc_miss = 0.5 # 50% of missing data
    # xmiss = np.copy(xfull)
    # xmiss_flat = xmiss.flatten()
    # miss_pattern = np.random.choice(n*p, np.floor(n*p*perc_miss).astype(np.int), replace=False)
    # xmiss_flat[miss_pattern] = np.nan
    # xmiss = xmiss_flat.reshape([n,p]) # in xmiss, the missing values are represented by nans
    # mask = np.isfinite(xmiss) # binary mask that indicates which values are missing
    # print(odata[0])
    n, p = odata.shape
    my_scaler = StandardScaler()
    my_scaler.fit(odata)
    mean_x, std_x = my_scaler.mean_, my_scaler.scale_
    xfull = my_scaler.transform(odata)
    xmiss =np.copy(xfull)
    mask = np.isfinite(data)

    xhat_0 = np.copy(xmiss)
    xhat_0[np.isnan(data)] = 0
    ## hyperparameters
    h = 128 # number of hidden units in (same for all MLPs)
    d = 1 # dimension of the latent space
    K = 20 # number of IS during training

    p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)

    decoder = nn.Sequential(
        torch.nn.Linear(d, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 3*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
    )

    encoder = nn.Sequential(
        torch.nn.Linear(p, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
    )

    encoder.cuda() # we'll use the GPU
    decoder.cuda()

    def miwae_loss(iota_x,mask):
        # print(iota_x[torch.isnan(iota_x)])
        iota_x[torch.isnan(iota_x)] = 0
        batch_size = iota_x.shape[0]
        out_encoder = encoder(iota_x)
        # print(out_encoder[..., :d])
        # print(torch.nn.Softplus()(out_encoder[..., d:(2*d)]))
        # q_zgivenxobs = td.Independent(
        #             td.Normal(loc=out_encoder[..., :d], scale=torch.nn.Softplus()(out_encoder[..., d:(2 * d)])), 1)

        q_zgivenxobs = td.Independent(
                td.Normal(loc=out_encoder[..., :d], scale=torch.nn.Softplus()(out_encoder[..., d:(2 * d)])), 1)
        zgivenx = q_zgivenxobs.rsample([K])
        zgivenx_flat = zgivenx.reshape([K*batch_size,d])

        out_decoder = decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :p]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3

        data_flat = torch.Tensor.repeat(iota_x,[K,1]).reshape([-1,1])
        tiledmask = torch.Tensor.repeat(mask,[K,1])

        all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K*batch_size,p])

        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([K,batch_size])
        logpz = p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))

        return neg_bound

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=1e-6)

    def miwae_impute(iota_x,mask,L):
        batch_size = iota_x.shape[0]
        out_encoder = encoder(iota_x)
        q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :d],scale=torch.nn.Softplus()(out_encoder[..., d:(2*d)])),1)

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L*batch_size,d])

        out_decoder = decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :p]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p:(2*p)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*p):(3*p)]) + 3

        data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).cuda()
        tiledmask = torch.Tensor.repeat(mask,[L,1]).cuda()

        all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,p])

        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
        logpz = p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

        imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L,batch_size,p])
        xm=torch.einsum('ki,kij->ij', imp_weights, xms)

        return xm

    def weights_init(layer):
        if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)


    miwae_loss_train=np.array([])
    mse_train=np.array([])
    mse_train2=np.array([])
    bs = 64 # batch size
    n_epochs = 30
    xhat = np.copy(xhat_0) # This will be out imputed data matrix

    encoder.apply(weights_init)
    decoder.apply(weights_init)

    for ep in range(1,n_epochs):
        print(ep)
        perm = np.random.permutation(n) # We use the "random reshuffling" version of SGD
        batches_data = np.array_split(xhat_0[perm, ], n/bs)
        batches_mask = np.array_split(mask[perm, ], n/bs)
        losses = []
        for it in range(len(batches_data)):
            optimizer.zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()
            b_data = torch.from_numpy(batches_data[it]).float().cuda()
            b_mask = torch.from_numpy(batches_mask[it]).float().cuda()
            loss = miwae_loss(iota_x = b_data,mask = b_mask)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(np.mean(losses))
        if ep % 5 == 0:
            print('Epoch %g' %ep)
            print('MIWAE likelihood bound  %g' %(-np.log(K)-miwae_loss(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda()).cpu().data.numpy())) # Gradient step

            ### Now we do the imputation
            xhat[~mask] = miwae_impute(iota_x = torch.from_numpy(xhat_0).float().cuda(),mask = torch.from_numpy(mask).float().cuda(),L=10).cpu().data.numpy()[~mask]
            # xhat = xhat * std_x[None, :] + mean_x[None, :]
            # return xhat
            err = np.array([mse(xhat,xfull,mask)])
            mse_train = np.append(mse_train,err,axis=0)
            print('Imputation MSE  %g' %err)
            print('-----')
    xhat = xhat * std_x[None, :] + mean_x[None, :]
    return xhat

if __name__ == "__main__":
    time_start = time.time()  # 开始计时
    print(time_start)
    # 要执行的代码，或函数
    # 要执行的代码，或函数


    miss_rate = 0.5
    dataset_name = "gpu"
    out = "/data/lsw/representations/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_" + str(miss_rate) + ".csv"
    full_out = "/data/lsw/representations/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_" + str(
        0.0) + ".csv"
    # data_name = "data/" + dataset_name + "/" + dataset_name + "_full.csv"
    # full_data_name = "data/" + dataset_name + "/" + dataset_name + "_full.csv"
    data = pd.read_csv(out)
    full_data = pd.read_csv(full_out)
    x_full = full_data.iloc[:, :-1]
    data_x = data.iloc[:, :-1]

    imputed_data = MIWAE(np.array(data_x), np.array(x_full))
    imputed_data = pd.DataFrame(imputed_data)
    data.iloc[:, :-1] = imputed_data
    imp_out = "/data/lsw/representations/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_" + str(
        miss_rate) + "_miwae.csv"
    data.to_csv(imp_out, index=None)
    time_end = time.time()  # 结束计时

    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')
