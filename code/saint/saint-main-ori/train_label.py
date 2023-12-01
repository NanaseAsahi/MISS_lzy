import json
import time
from pathlib import Path

import torch
from torch import nn
from models import SAINT

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon,data_prep_openml_label
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask
from augmentations import add_noise
from early_stopping import EarlyStopping
import os
import numpy as np

def run(runs, dataset_name, missingrate, task, missingtype, imp):
    print(best_test_accuracy_list)
    starttime = time.time()
    modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, task, str(dataset_name), str(missingrate),
                                  opt.run_name, str(run))
    print("model", modelsave_path)
    if task == 'regression':
        opt.dtask = 'reg'
    else:
        opt.dtask = 'clf'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")

    torch.manual_seed(opt.set_seed[runs])
    os.makedirs(modelsave_path, exist_ok=True)

    if opt.active_log:
        import wandb
        if opt.pretrain:
            wandb.init(project="saint_v2_all", group=opt.run_name,
                       name=f'pretrain_{task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed[runs])}')
        else:
            if opt.task == 'multiclass':
                wandb.init(project="saint_v2_all_kamal", group=opt.run_name,
                           name=f'{task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed[runs])}')
            else:
                wandb.init(project="saint_v2_all", group=opt.run_name,
                           name=f'{task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed[runs])}')

    print('Downloading and processing the dataset, it might take some time.')
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(
        dataset_name, missingrate, opt.dset_seed[runs], task,

        [.8, .1, .1], missingtype, imp)
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    ##### Setting some hyperparams based on inputs and dataset
    _, nfeat = X_train['data'].shape
    if nfeat > 100:
        opt.embedding_size = min(8, opt.embedding_size)
        opt.batchsize = min(64, opt.batchsize)
    if opt.attentiontype != 'col':
        opt.transformer_depth = 1
        opt.attention_heads = min(4, opt.attention_heads)
        opt.attention_dropout = 0.8
        opt.embedding_size = min(32, opt.embedding_size)
        opt.ff_dropout = 0.8

    print(nfeat, opt.batchsize)
    print(opt)

    if opt.active_log:
        wandb.config.update(opt)
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)

    valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

    test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
    if task == 'regression':
        y_dim = 1
    else:
        y_dim = 2

    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(
        int)  # Appending 1 for CLS token, this is later used to generate embeddings.

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

    if  task == 'binary':
        # opt.task = 'binary'
        criterion = nn.CrossEntropyLoss().to(device)
    elif  task == 'multiclass':
        # opt.task = 'multiclass'
        criterion = nn.CrossEntropyLoss().to(device)
    elif task == 'regression':
        criterion = nn.MSELoss().to(device)
    else:
        raise 'case not written yet'

    model.to(device)

    if opt.pretrain:
        from pretraining import SAINT_pretrain
        model = SAINT_pretrain(model, cat_idxs, X_train, y_train, continuous_mean_std, opt, device, modelsave_path)
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml_label(
        dataset_name, missingrate, opt.dset_seed[runs], task,

        [.8, .1, .1], missingtype, imp)
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=0)

    valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
    validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=0)

    test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=0)
    ## Choosing the optimizer
    ckpt_path = modelsave_path + "/models/early"
    early_stopping = EarlyStopping(ckpt_path)
    state_dict = torch.load(ckpt_path + "/best_network.pth")
    model.load_state_dict(state_dict)
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                              momentum=0.9, weight_decay=5e-4)
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

    train_loss = []
    val_acc = []
    test_acc = []
    for epoch in range(opt.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont.
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), \
                                                         data[3].to(device), data[4].to(device)

            # We are converting the data to embeddings in the next step
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:, 0, :]

            y_outs = model.mlpfory(y_reps)
            if task == 'regression':
                loss = criterion(y_outs, y_gts)
            else:
                y_gts = y_gts.to(dtype=torch.long)
                loss = criterion(y_outs, y_gts.squeeze())
            loss.backward()
            optimizer.step()
            if opt.optimizer == 'SGD':
                scheduler.step()
            running_loss += loss.item()
        # print(running_loss)
        train_loss.append(running_loss)
        if opt.active_log:
            wandb.log({'epoch': epoch, 'train_epoch_loss': running_loss,
                       'loss': loss.item()
                       })
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                if task in ['binary', 'multiclass']:
                    accuracy, auroc = classification_scores(model, validloader, device, task, vision_dset)
                    test_accuracy, test_auroc = classification_scores(model, testloader, device, task, vision_dset)
                    val_acc.append(accuracy.item())
                    test_acc.append(test_accuracy.item())
                    print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                          (epoch + 1, accuracy, auroc))
                    print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                          (epoch + 1, test_accuracy, test_auroc))
                    if opt.active_log:
                        wandb.log({'valid_accuracy': accuracy, 'valid_auroc': auroc})
                        wandb.log({'test_accuracy': test_accuracy, 'test_auroc': test_auroc})
                    if task == 'multiclass':
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy
                            torch.save(model.state_dict(), '%s/finalbestmodel.pth' % (modelsave_path))
                    else:
                        if accuracy > best_valid_accuracy:
                            best_valid_accuracy = accuracy
                            # if auroc > best_valid_auroc:
                            #     best_valid_auroc = auroc
                            best_test_auroc = test_auroc
                            best_test_accuracy = test_accuracy
                            torch.save(model.state_dict(), '%s/finalbestmodel.pth' % (modelsave_path))

                else:
                    valid_rmse, valid_R2 = mean_sq_error(model, validloader, device, vision_dset)
                    test_rmse, test_R2 = mean_sq_error(model, testloader, device, vision_dset)
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
            break  # 跳出迭代，结束训练

    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))
    if task == 'binary':
        print('AUROC on best model:  %.3f' % (best_test_auroc))
        print('Accuracy on best model test:  %.3f' % (best_test_accuracy))
        best_test_auroc_list.append(best_test_auroc.item())
        print(best_test_accuracy_list)
        best_test_accuracy_list.append(best_test_accuracy.item())
    elif task == 'multiclass':
        print('AUROC on best model:  %.3f' % (best_test_auroc))
        print('Accuracy on best model test:  %.3f' % (best_test_accuracy))
        best_test_auroc_list.append(best_test_auroc.item())
        best_test_accuracy_list.append(best_test_accuracy.item())
    else:
        print('RMSE on best model:  %.3f' % (best_test_rmse))
        print('R2 on best model:  %.3f' % (best_test_R2))
        best_test_auroc_list.append(best_test_rmse)
        best_test_accuracy_list.append(best_test_R2)
    print("time", time.time() - starttime)
    if opt.active_log:
        if task == 'regression':
            wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep': best_test_rmse,
                       'cat_dims': len(cat_idxs), 'con_dims': len(con_idxs)})
        else:
            wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep': best_test_auroc,
                       'test_accuracy_bestep': best_test_accuracy, 'cat_dims': len(cat_idxs),
                       'con_dims': len(con_idxs)})
    import matplotlib.pyplot as plt

    # 横轴数据

    # 绘制线条
    # print(train_loss)
    # plt.plot(np.arange(len(train_loss)), train_loss)
    # plt.plot(np.arange(len(val_acc)), val_acc)
    # # 显示图形
    # plt.show()
    return best_test_auroc_list, best_test_accuracy_list

seeds = [0]
imputations = ["_missforest"]
missingrates = [0.1, 0.3, 0.5, 0.7, 0.9]
mul_datasets = []
reg_datasets = ["temperature"]
bi_datasets = ["News"]
missingtypes =  ["mcar_label_"]
# imputations = ["mean", "_notmiwae", "_miwae", "_gain", "_missforest"]
# bi_datasets = ["HI", "News"]
# reg_datasets = ["temperature", "gas"]
# mul_datasets = ['Letter', 'Gesture']
parser = argparse.ArgumentParser()

parser.add_argument('--data_name', required=False, type=str)
parser.add_argument('--missing_rate', required=False, type=float)
parser.add_argument('--vision_dset', action='store_true')
parser.add_argument('--task', required=False, type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='col', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default=[0, 149669, 52983, 746806, 639519], type=int)
parser.add_argument('--dset_seed', default=[0, 149669, 52983, 746806, 639519], type=int)
parser.add_argument('--active_log', action = 'store_true')

parser.add_argument('--pretrain', default=True)
parser.add_argument('--pretrain_epochs', default=1, type=int)
parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
parser.add_argument('--pt_aug', default=[], type=str, nargs='*',choices = ['mixup','cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)

parser.add_argument('--train_mask_prob', default=0, type=float)
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default= 0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])

best_valid_accuracy_list = []
best_test_accuracy_list = []
opt = parser.parse_args()
t = time.localtime()
t_time = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + str(t.tm_min)
all_result = []
for data in bi_datasets:
    for missingrate in missingrates:
        for missingtype in missingtypes:
            for imp in imputations:
                best_valid_accuracy_list = []
                best_test_accuracy_list = []
                best_test_auroc_list = []
                time_list = []

                for runs in range(1):
                    starttime = time.time()
                    best_test_auroc_list, best_test_accuracy_list = run(runs, data, missingrate,
                                                                                                  "binary", missingtype, imp)
                    time_list.append(time.time() - starttime)
                result = {"data": data, "missingrate": missingrate, "task": "binary", "type": missingtype,
                          "time": time_list,
                              "seed": seeds,
                          "imp": imp,
                          "best_test_auroc_list": best_test_auroc_list,
                          "best_test_accuracy_list": best_test_accuracy_list,}
                # ,"pretrain_epoch": pretrain_epoch,"pretrain_loss_con": pretrain_loss_con,"pretrain_loss_deno": pretrain_loss_deno,"pretrain_loss": pretrain_loss}

                out = Path("/data/lsw/result/saint_ori_allresult_" + t_time + ".json")
                with open(out, "a") as f:
                        json.dump(result, f, indent=4)

for data in mul_datasets:
    for missingrate in missingrates:
        for missingtype in missingtypes:
            for imp in imputations:
                best_valid_accuracy_list = []
                best_test_accuracy_list = []
                best_test_auroc_list = []
                time_list = []
                for runs in range(1):
                    starttime = time.time()
                    best_test_auroc_list, best_test_accuracy_list = run(runs, data, missingrate,
                                                                                                  "multiclass", missingtype, imp)
                    time_list.append(time.time() - starttime)
                result = {"data": data, "missingrate": missingrate, "task": "multiclass", "type": missingtype,
                          "time": time_list,
                              "seed": seeds,
                          "imp": imp,
                          "best_test_auroc_list": best_test_auroc_list,
                          "best_test_accuracy_list": best_test_accuracy_list,}
            # ,"pretrain_epoch": pretrain_epoch,"pretrain_loss_con": pretrain_loss_con,"pretrain_loss_deno": pretrain_loss_deno,"pretrain_loss": pretrain_loss}

                out = Path("/data/lsw/result/saint_ori_allresult_" + t_time + ".json")
                with open(out, "a") as f:
                        json.dump(result, f, indent=4)

for data in reg_datasets:
    for missingrate in missingrates:
        for missingtype in missingtypes:
            for imp in imputations:
                best_valid_accuracy_list = []
                best_test_accuracy_list = []
                best_test_auroc_list = []
                time_list = []
                for runs in range(1):
                    starttime = time.time()
                    best_test_auroc_list, best_test_accuracy_list = run(runs, data, missingrate,
                                                                                                  "regression", missingtype, imp)
                    time_list.append(time.time() - starttime)
                result = {"data": data, "missingrate": missingrate, "task": "regression", "type": missingtype,
                          "time": time_list,
                              "seed": seeds,
                          "imp": imp,
                          "best_test_rmse_list": best_test_auroc_list,
                          "best_test_R2_list": best_test_accuracy_list,}
                out = Path("/data/lsw/result/saint_ori_allresult_" + t_time + ".json")
                with open(out, "a") as f:
                        json.dump(result, f, indent=4)