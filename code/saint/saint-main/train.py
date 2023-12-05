import time
from pathlib import Path

import torch
from torch import nn
from models import SAINT

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise
from early_stopping import EarlyStopping
import os
import numpy as np
import json
import pandas as pd
from IPS import sampling, compute_ips
#设定缺失率及数据集
missingrates = [0.1, 0.3, 0.5, 0.7, 0.9]
mul_datasets = ["gas"]
reg_datasets = ["temperature"]
bi_datasets = ["HI", "News"]
missingtypes = ["mcar_", "mar_p_", "mnar_p_"]

ips_nums = [40]
seeds = [0, 149669, 52983]
best_valid_accuracy_list = []
best_test_accuracy_list = []
best_test_auroc_list = []
col_softmax = nn.Softmax(dim=1)
row_softmax = nn.Softmax(dim=0)

criterion2 = nn.MSELoss(reduction='none')
criterion3 = nn.LogSoftmax(dim=1)
def run(runs, dataset_name, missingrate, task, missingtype, ips_num):
    starttime = time.time()
    datafull = pd.read_csv(
        Path("/data/lsw/data/data/" + dataset_name + "/" + "mcar_" + dataset_name + "_" + str(0.0) + ".csv"))
    datamiss = pd.read_csv(Path(
        "/data/lsw/data/data/" + dataset_name + "/" + missingtype + dataset_name + "_" + str(missingrate) + ".csv"))
    # 标记专家数据
    datamiss, indicator = sampling(datafull, datamiss, ips_num, method='feature')
    ips_start = time.time()
    ips = compute_ips(datamiss[:, :-1], indicator[:, :-1], num=ips_num, method='xgb')
    ips = torch.tensor(ips)
    row_ips = torch.sum(ips, dim=1)
    row_ips = row_softmax(row_ips)
    ips = col_softmax(ips)
    print("ips_num:" ,ips_num, "time:", time.time() - ips_start)
    modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, task, dataset_name, str(missingrate), opt.run_name, str(run))
    if task == 'regression':
        opt.dtask = 'reg'
    else:
        opt.dtask = 'clf'


    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # print(f"Device is { device}.")

    torch.manual_seed(opt.set_seed[runs])
    os.makedirs(modelsave_path, exist_ok=True)
    #对数据进行切割及处理
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(dataset_name, missingrate, opt.dset_seed[runs], task, [.8, .1, .1], ips, row_ips, missingtype)

    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    ##### Setting some hyperparams based on inputs and dataset
    _,nfeat = X_train['data'].shape
    if nfeat > 100:
        opt.embedding_size = min(8,opt.embedding_size)
        opt.batchsize = min(64, opt.batchsize)
    if opt.attentiontype != 'col':
        opt.transformer_depth = 1
        opt.attention_heads = min(4,opt.attention_heads)
        opt.attention_dropout = 0.8
        opt.embedding_size = min(32,opt.embedding_size)
        opt.ff_dropout = 0.8
        pass

    print(nfeat, opt.batchsize)
    print(opt)
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=0)

    valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

    test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
    if task == 'regression':
        y_dim = 1
    else:
        y_dim = len(np.unique(y_train['data'][:, 0]))

    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.

    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=opt.embedding_size,
        dim_out=1,
        depth=opt.transformer_depth,
        heads=opt.attention_heads,
        attn_dropout=opt.attention_dropout,
        ff_dropout=opt.ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=opt.cont_embeddings,
        attentiontype=opt.attentiontype,
        final_mlp_style=opt.final_mlp_style,
        y_dim=y_dim
    )
    vision_dset = opt.vision_dset

    #确定loss函数
    if y_dim == 2 and task == 'binary':
        criterion = nn.CrossEntropyLoss().to(device)
    elif y_dim > 2 and task == 'multiclass':
        criterion = nn.CrossEntropyLoss().to(device)
    elif task == 'regression':
        criterion = nn.MSELoss().to(device)
    else:
        raise'case not written yet'

    model.to(device)

    # 进行预训练
    ckpt_path = modelsave_path + "/models/early"
    try:
        os.makedirs(ckpt_path, exist_ok=False)
    except:
        try:
            os.remove(ckpt_path + "/best_network.pth")
        except:
            pass

    ckpt_path = modelsave_path + "/models/early"
    early_stopping = EarlyStopping(ckpt_path)
    pretrain_start_time = time.time()
    if opt.pretrain:
        from pretraining import SAINT_pretrain

        model, pretrain_epoch, pretrain_loss_con, pretrain_loss_deno, pretrain_loss = SAINT_pretrain(model, cat_idxs, X_train, y_train, continuous_mean_std, opt, device, modelsave_path ,opt.attentiontype, datafull)

        # ckpt_path = "/data/lsw/saint/saint-main/bestmodels/binary/News/0.5/testrun/<function run at 0x7efff6549310>/models/early"

        state_dict = torch.load(ckpt_path + "/best_network.pth")
        model.load_state_dict(state_dict)
    pretrain_end_time = time.time()
    ## Choosing the optimizer
    #读取预训练模型参数

    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
        from utils import get_scheduler
        scheduler = get_scheduler(opt, optimizer)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    best_valid_auroc = 0
    best_valid_accuracy = 0
    best_test_auroc = 0
    best_test_accuracy = 0
    best_test_R2 = 0
    best_valid_rmse = 100000
    print('Training begins now.')

    # print(model)
    train_loss = []
    val_acc = []
    test_acc = []
    #进行训练
    # This sets requires_grad to False for all parameters without the string "lora_" in their names
    for epoch in range(1000):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont.
            x_categ, x_cont, y_gts, cat_mask, con_mask, cat_ips, con_ips, row_ips = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)
            # We are converting the data to embeddings in the next step
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
            if opt.attentiontype == 'col':
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, row_ips)
            # if opt.attentiontype == 'col':
            #     cat_outs, con_outs = model(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
            # else:
            #     cat_outs, con_outs = model(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, row_ips)
            # reps = model.transformer(x_categ_enc, x_cont_enc)
            # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:,0,:]
            cat_outs = model.mlp1(reps[:, :model.num_categories, :])
            con_outs = model.mlp2(reps[:, model.num_categories:, :])
            y_outs = model.mlpfory(y_reps)
            if task == 'regression':
                loss = criterion(y_outs,y_gts)
            else:
                loss = criterion(y_outs,y_gts.squeeze())
            if len(con_outs) > 0:
                con_outs = torch.cat(con_outs, dim=1)
                l2 = criterion2(con_outs, x_cont)
                l2[con_mask == 0] = 0
                # l2 = l2 * con_ips
                l2 = l2.mean()
                # print(l2)
            else:
                l2 = 0
            l1 = 0
            # import ipdb; ipdb.set_trace()
            n_cat = x_categ.shape[-1]
            # print(cat_outs,len(cat_outs))
            # print(x_categ,x_categ.shape)
            reconstruction_errors_cat = torch.zeros(x_categ.shape).to(x_categ.device)
            for j in range(1, n_cat):
                log_x = criterion3(cat_outs[j])
                log_x = log_x[range(cat_outs[j].shape[0]), x_categ[:, j]]
                log_x[cat_mask[:, j] == 0] = 0
                # log_x *= cat_ips[:, j]
                l1 += abs(sum(log_x) / cat_outs[j].shape[0])
                # l1 += criterion1(cat_outs[j], x_categ[:, j])
            # print(loss, l1, l2)
            # loss += opt.lam2 * l1 + opt.lam3 * l2
            loss.backward()
            optimizer.step()
            if opt.optimizer == 'SGD':
                scheduler.step()
            running_loss += loss.item()
        # print(running_loss)
        train_loss.append(running_loss)
        if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    if task in ['binary', 'multiclass']:
                        accuracy, auroc = classification_scores(model, validloader, device, task, vision_dset, opt.attentiontype)
                        test_accuracy, test_auroc = classification_scores(model, testloader, device, task, vision_dset, opt.attentiontype)
                        val_acc.append(accuracy.item())
                        test_acc.append(test_accuracy.item())
                        print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                            (epoch + 1, accuracy,auroc ))
                        print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                            (epoch + 1, test_accuracy,test_auroc ))
                        # if opt.active_log:
                        #     wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })
                        #     wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })
                        if task =='multiclass':
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                                best_test_auroc = test_auroc
                                # print('%s/bestmodel.pth' % (modelsave_path))
                                torch.save(model.state_dict(), '%s/finalbestmodel.pth' % (modelsave_path))
                            if best_test_accuracy < test_accuracy:
                                best_test_accuracy = test_accuracy
                                best_test_auroc = test_auroc
                        else:
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                            # if auroc > best_valid_auroc:
                            #     best_valid_auroc = auroc
                                torch.save(model.state_dict(),'%s/finalbestmodel.pth' % (modelsave_path))
                            if best_test_accuracy < test_accuracy:
                                best_test_accuracy = test_accuracy
                                best_test_auroc = test_auroc

                    else:
                        valid_rmse, valid_R2 = mean_sq_error(model, validloader, device,vision_dset)
                        test_rmse, test_R2 = mean_sq_error(model, testloader, device,vision_dset)
                        print('[EPOCH %d] VALID RMSE: %.3f, VALID R2: %.3f' %
                              (epoch + 1, valid_rmse, valid_R2))
                        print('[EPOCH %d] TEST RMSE: %.3f, VALID R2: %.3f' %
                              (epoch + 1, test_rmse, test_R2))
                        # if opt.active_log:
                        #     wandb.log({'valid_rmse': valid_rmse ,'test_rmse': test_rmse })
                        if valid_rmse < best_valid_rmse:
                            best_valid_rmse = valid_rmse
                            best_test_rmse = test_rmse
                            best_test_R2 = test_R2
                            torch.save(model.state_dict(), '%s/finalbestmodel.pth' % (modelsave_path))
                model.train()
        try:
            early_stopping(100 - accuracy, model)
        except:
            early_stopping(valid_rmse, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
    if task =='binary':
        print('AUROC on best model:  %.3f' %(best_test_auroc))
        print('Accuracy on best model test:  %.3f' % (best_test_accuracy))
        best_test_auroc_list.append(best_test_auroc.item())
        best_test_accuracy_list.append(best_test_accuracy.item())
    elif task =='multiclass':
        print('AUROC on best model:  %.3f' % (best_test_auroc))
        print('Accuracy on best model test:  %.3f' %(best_test_accuracy))
        best_test_auroc_list.append(best_test_auroc.item())
        best_test_accuracy_list.append(best_test_accuracy.item())
    else:
        print('RMSE on best model:  %.3f' %(best_test_rmse))
        print('R2 on best model:  %.3f' % (best_test_R2))
        best_test_auroc_list.append(best_test_rmse)
        best_test_accuracy_list.append(best_test_R2)
    print("pretrain_time", pretrain_end_time - pretrain_start_time)
    print("cost_time", time.time() - starttime)
        # if opt.active_log:
    #     if opt.task == 'regression':
    #         wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep':best_test_rmse ,
    #         'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })
    #     else:
    #         wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc ,
    #         'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })
    best_valid_accuracy = float(best_valid_accuracy)
    best_test_accuracy = float(best_test_accuracy)
    # result = {"best_accuracy_val": float(best_valid_accuracy), "best_accuracy_test": float(best_test_accuracy),
    #           "train_loss": train_loss, "val_acc": val_acc, "test_acc": test_acc,
    #           "best_test_accuracy": np.mean(best_test_accuracy_list),
    #           "best_valid_accuracy": np.mean(best_valid_accuracy_list)}
    #           # ,"pretrain_epoch": pretrain_epoch,"pretrain_loss_con": pretrain_loss_con,"pretrain_loss_deno": pretrain_loss_deno,"pretrain_loss": pretrain_loss}
    # out = Path("/home/tuser1/result/" + dataset_name + "/" + missingtype + dataset_name + str(missingrate) + ".json")
    #
    # with open(out, "w") as f:
    #     json.dump(result, f, indent=4)
    # import matplotlib.pyplot as plt
    #
    # plt.plot(np.arange(len(train_loss)), train_loss)
    # plt.plot(np.arange(len(val_acc)), val_acc)
    # plt.show()
    return best_test_auroc_list, best_test_accuracy_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # 设定参数
    # parser.add_argument('--data_name', required=True, type=str)
    # parser.add_argument('--missing_rate', required=True, type=float)
    parser.add_argument('--vision_dset', action='store_true')
    # parser.add_argument('--task', required=True, type=str, choices=['binary', 'multiclass', 'regression'])
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--c', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='col', type=str,
                        choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])

    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--savemodelroot', default='bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=[0, 149669, 52983, 746806, 639519], type=int)
    parser.add_argument('--dset_seed', default=[0, 149669, 52983, 746806, 639519], type=int)
    parser.add_argument('--active_log', action='store_true')

    parser.add_argument('--pretrain', default=True)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*',
                        choices=['contrastive', 'denoising', 'mask']) #选择预训练模式
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)

    parser.add_argument('--ssl_avail_y', default=0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)

    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--lam4', default=1, type=float)
    parser.add_argument('--lam5', default=1, type=float)
    parser.add_argument('--pretrain_ratio', default=0.5, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
    t = time.localtime()
    t_time = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + str(t.tm_min)
    opt = parser.parse_args()
    all_result = []
    for data in bi_datasets:
        for missingrate in missingrates:
            for missingtype in missingtypes:
                for ips_num in ips_nums:
                    best_valid_accuracy_list = []
                    best_test_accuracy_list = []
                    best_test_auroc_list = []
                    time_list = []
                    for runs in range(1):
                        starttime = time.time()
                        best_test_auroc_list, best_test_accuracy_list = run(runs, data, missingrate, "binary", missingtype, ips_num)
                        print(best_test_accuracy_list)
                        time_list.append(time.time() - starttime)
                        print( "time", time.time() - starttime)
                    result = {"data": data, "missingrate": missingrate, "task": "binary", "type": missingtype, "ips_num": ips_num,
                              "time": time_list,
                              "seed": seeds,
                              "best_test_auroc_list": best_test_auroc_list,
                              "best_test_accuracy_list": best_test_accuracy_list,}

                    out = Path("/data/lsw/result/saint_allresult_" + t_time + ".json")
                    with open(out, "a") as f:
                        json.dump(result, f, indent=4)
                        f.write("\n")

    for data in mul_datasets:
        for missingrate in missingrates:
            for missingtype in missingtypes:
                for ips_num in ips_nums:
                    best_valid_accuracy_list = []
                    best_test_accuracy_list = []
                    best_test_auroc_list = []
                    time_list = []
                    for runs in range(1):
                        starttime = time.time()
                        best_test_auroc_list, best_test_accuracy_list = run(runs, data, missingrate, "multiclass", missingtype, ips_num)
                        time_list.append(time.time() - starttime)
                    result = {"data": data, "missingrate": missingrate, "task": "multiclass", "type": missingtype, "ips_num": ips_num,
                              "time": time_list,
                              "seed": seeds,
                              "best_test_auroc_list": best_test_auroc_list,
                              "best_test_accuracy_list": best_test_accuracy_list,}

                    out = Path("/data/lsw/result/saint_allresult_" + t_time + ".json")
                    with open(out, "a") as f:
                            json.dump(result, f, indent=4)
                            f.write("\n")
    for data in reg_datasets:
        for missingrate in missingrates:
            for missingtype in missingtypes:
                for ips_num in ips_nums:
                    best_valid_accuracy_list = []
                    best_test_accuracy_list = []
                    best_test_auroc_list = []
                    time_list = []
                    for runs in range(1):
                        starttime = time.time()
                        best_test_auroc_list, best_test_accuracy_list = run(runs, data,
                                                                                                      missingrate,
                                                                                                      "regression",
                                                                                                      missingtype, ips_num)
                        time_list.append(time.time() - starttime)
                    result = {"data": data, "missingrate": missingrate, "task": "regression", "type": missingtype, "ips_num": ips_num,
                              "time": time_list,
                              "seed": seeds,
                              "best_test_rmse_list": best_test_auroc_list,
                              "best_test_R2_list": best_test_accuracy_list,}

                    out = Path("/data/lsw/result/saint_allresult_" + t_time + ".json")
                    with open(out, "a") as f:
                            json.dump(result, f, indent=4)
                            f.write("\n")