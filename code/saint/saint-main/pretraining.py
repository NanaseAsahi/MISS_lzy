import torch
from torch import nn

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
from torch.utils.data import DataLoader
import torch.optim as optim
from augmentations import embed_data_mask
from augmentations import add_noise
from early_stopping import EarlyStopping
import os
import numpy as np
from IPS import sampling, compute_ips
import pandas as pd

#模型的预训练模块
def SAINT_pretrain(model, cat_idxs, X_train, y_train,continuous_mean_std,opt,device,model_save_path,attention_type, datafull):
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
    vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(),lr=0.0001)
    pt_aug_dict = {
        'noise_type': opt.pt_aug,
        'lambda': opt.pt_aug_lam
    }
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.LogSoftmax(dim=1)
    print("Pretraining begins!")
    # 设定earlystop
    ckpt_path = model_save_path + "/models/early"
    try:
        os.makedirs(ckpt_path, exist_ok=False)
    except:
        try:
            os.remove(ckpt_path + "/best_network.pth")
        except:
            pass
    early_stopping = EarlyStopping(ckpt_path, patience=3)
    pretrain_epoch = 0
    pretrain_loss_con = []
    pretrain_loss_deno = []
    pretrain_loss = []
    for epoch in range(opt.pretrain_epochs):
        pretrain_epoch += 1
        model.train()
        c_loss = 0.0
        m_loss = 0.0
        d_loss_con = 0.0
        d_loss_cat = 0.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_categ, x_cont, _, cat_mask, con_mask, cat_ips, con_ips, rowips = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device)
            # cate,con bs*feature
            # print("x_cont", x_cont, x_cont.shape)
            # print("x_categ", x_categ, x_categ.shape)
            # print("mask", con_mask, con_mask.shape)
            # embed_data_mask function is used to embed both categorical and continuous data.
            if 'cutmix' in opt.pt_aug:
                from augmentations import add_noise
                x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model,vision_dset)
            else:
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            # cate,con bs*feature*32
            # print()
            # print("x_categ_enc_2,x_cont_enc_2:", x_categ_enc_2, x_cont_enc_2)
            # print("x_categ_enc,x_cont_enc:", x_categ_enc.shape, x_cont_enc.shape)
            if 'mixup' in opt.pt_aug:
                from augmentations import mixup_data
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=opt.mixup_lam)
            loss = 0
            # contrastive pretrain模块
            if 'contrastive' in opt.pt_tasks:
                # print(mask.shape)
                if attention_type == 'col':
                    aug_features_1 = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
                    aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2, con_mask, cat_mask, cat_ips, con_ips)
                else:
                    aug_features_1 = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, rowips)
                    aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2, con_mask, cat_mask, cat_ips,
                                                       con_ips, rowips)
                # print(aug_features_1.shape)
                # aug_features_1 = model.transformer(x_categ_enc, x_cont_enc)
                # aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                # print(aug_features_1.shape)
                if opt.pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif opt.pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')
                # print(aug_features_1.shape)
                # print(aug_features_2.shape)
                logits_per_aug1 = aug_features_1 @ aug_features_2.t()/opt.nce_temp
                logits_per_aug2 = aug_features_2 @ aug_features_1.t()/opt.nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                # print(logits_per_aug1.shape)
                # print(targets)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                c_loss += (loss_1 + loss_2)/2
                loss = opt.lam0 * (loss_1 + loss_2)/2
                pretrain_loss_con.append(loss.item())
            # denosing 模块
            if 'denoising' in opt.pt_tasks:
                mask = con_mask
                # print(mask.shape)
                if attention_type == 'col':
                    cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2, con_mask, cat_mask, cat_ips, con_ips)
                else:
                    cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2, con_mask, cat_mask, cat_ips, con_ips,rowips)
                if len(con_outs) > 0:
                    con_outs = torch.cat(con_outs, dim=1)
                    l2 = criterion2(con_outs[con_mask == 1], x_cont[con_mask == 1])
                    # print(l2)
                else:
                    l2 = 0
                l1 = 0
                # import ipdb; ipdb.set_trace()
                n_cat = x_categ.shape[-1]
                # print(cat_outs,len(cat_outs))
                # print(x_categ,x_categ.shape)
                reconstruction_errors_cat = torch.zeros(x_categ.shape).to(x_categ.device)
                for j in range(1,n_cat):
                    log_x = criterion3(cat_outs[j])
                    log_x = log_x[range(cat_outs[j].shape[0]), x_categ[:, j]]
                    log_x[cat_mask[:, j] == 0] = 0
                    l1 += abs(sum(log_x) / cat_outs[j].shape[0])
                    # l1 += criterion1(cat_outs[j], x_categ[:, j])
                d_loss_con += l1
                d_loss_cat += l2
                t_loss = opt.lam2 * l1 + opt.lam3 * l2
                loss += opt.lam2 * l1 + opt.lam3 * l2
                pretrain_loss_deno.append(t_loss.item())
                pretrain_loss.append(loss.item())
            # mask pretrain 模块
            if "mask" in opt.pt_tasks:
                # where =1 mask
                obfuscated_groups_cont = torch.bernoulli(
                    opt.pretrain_ratio * torch.ones((x_cont.shape), device=x_cont_enc.device)
                )
                obfuscated_groups_cat = torch.bernoulli(
                    opt.pretrain_ratio * torch.ones((x_categ.shape), device=x_categ_enc.device)
                )
                masked_x_cont = torch.mul(1 - obfuscated_groups_cont, x_cont)
                obfuscated_vars_cont = (1 - obfuscated_groups_cont).int()
                masked_x_cat = torch.mul(1 - obfuscated_groups_cat, x_categ).int()
                obfuscated_vars_cat = (1 - obfuscated_groups_cat).int()

                _, x_categ_enc_m, x_cont_enc_m = embed_data_mask(masked_x_cat, masked_x_cont, obfuscated_vars_cat, obfuscated_vars_cont, model, vision_dset)
                if attention_type == 'col':
                    cat_outs_m, con_outs_m = model(x_categ_enc_m, x_cont_enc_m, con_mask, cat_mask, cat_ips, con_ips)
                else:
                    cat_outs_m, con_outs_m = model(x_categ_enc_m, x_cont_enc_m, con_mask, cat_mask, cat_ips, con_ips, rowips)
                n_cat = x_categ.shape[-1]
                reconstruction_errors_cat = torch.zeros(x_categ.shape).to(x_categ.device)
                l_cate = 0
                for j in range(1, n_cat):
                    log_x = criterion3(cat_outs_m[j])
                    log_x = log_x[range(cat_outs_m[j].shape[0]), x_categ[:, j]]
                    log_x[cat_mask[:, j] == 0] = 0
                    log_x[obfuscated_groups_cat[:, j] == 0] = 0
                    reconstruction_errors_cat[:, j] = abs(log_x)

                batch_means_cat = torch.mean(x_categ.float(), dim=0)
                batch_means_cat[batch_means_cat == 0] = 1

                batch_stds_cat = torch.std(x_categ.float(), dim=0) ** 2
                batch_stds_cat[batch_stds_cat == 0] = batch_means_cat[batch_stds_cat == 0]
                features_loss_cat = torch.matmul(reconstruction_errors_cat, 1 / batch_stds_cat)
                nb_reconstructed_variables_cat = torch.sum(obfuscated_groups_cat, dim=1)
                eps = 1e-9
                features_loss_cat = features_loss_cat / (nb_reconstructed_variables_cat + eps)
                con_outs_m = torch.cat(con_outs_m, dim=1)
                errors_cont = con_outs_m - x_cont
                errors_cont[con_mask == 0] = 0
                reconstruction_errors_cont = torch.mul(errors_cont, obfuscated_groups_cont) ** 2
                batch_means_cont = torch.mean(x_cont, dim=0)
                batch_means_cont[batch_means_cont == 0] = 1

                batch_stds_cont = torch.std(x_cont, dim=0) ** 2
                batch_stds_cont[batch_stds_cont == 0] = batch_means_cont[batch_stds_cont == 0]
                features_loss_cont = torch.matmul(reconstruction_errors_cont, 1 / batch_stds_cont)
                nb_reconstructed_variables_cont = torch.sum(obfuscated_groups_cont, dim=1)
                eps = 1e-9
                features_loss_cont = features_loss_cont / (nb_reconstructed_variables_cont + eps)
                m_loss += (opt.lam4 * torch.mean(features_loss_cont)) + opt.lam5 * torch.mean(features_loss_cat)
                loss += (opt.lam4 * torch.mean(features_loss_cont)) + opt.lam5 * torch.mean(features_loss_cat)


            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        early_stopping(running_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
        print(f'Epoch: {epoch}, Running Loss: {running_loss}, contrastive Loss: {c_loss}, denosing Loss: {d_loss_con, d_loss_cat}, mask Loss: {m_loss}')

    print('END OF PRETRAINING!')
    return model, pretrain_epoch, pretrain_loss_con, pretrain_loss_deno, pretrain_loss
        # if opt.active_log:
        #     wandb.log({'pt_epoch': epoch ,'pretrain_epoch_loss': running_loss
        #     })
