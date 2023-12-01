import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import os
import numpy as np

def tab_ori_pretrain(model, args, X, batch_size, cat_idxs,X_train,y_train,continuous_mean_std,opt,device,model_save_path):
    train_dataset = TensorDataset(X)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2)
    pretrain_epochs = 100
    for epoch in range(pretrain_epochs):
        for i, batch_X in enumerate(train_loader):

            if args.cat_idx:
                x_categ = batch_X[:, args.cat_idx].int().to(self.device)
            else:
                x_categ = None

            x_cont = batch_X[:, self.num_idx].to(self.device)
            # print(x_categ, x_cont)
            out = self.model(x_categ, x_cont)

            if self.args.objective == "regression" or self.args.objective == "binary":
                out = out.squeeze()

            loss = loss_func(out, batch_y.to(self.device))
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)
    vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(),lr=0.0001)
    pt_aug_dict = {
        'noise_type' : opt.pt_aug,
        'lambda' : opt.pt_aug_lam
    }
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    print("Pretraining begins!")
    ckpt_path = model_save_path + "/models/early"
    try:
        os.makedirs(ckpt_path, exist_ok=False)
    except:
        try:
            os.remove(ckpt_path + "/best_network.pth")
        except:
            pass
    print(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)
    early_stopping = EarlyStopping(ckpt_path)
    for epoch in range(opt.pretrain_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_categ, x_cont, _ ,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)

            # embed_data_mask function is used to embed both categorical and continuous data.
            if 'cutmix' in opt.pt_aug:
                from augmentations import add_noise
                x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model,vision_dset)
            else:
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            
            if 'mixup' in opt.pt_aug:
                from augmentations import mixup_data
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=opt.mixup_lam)
            loss = 0
            if 'contrastive' in opt.pt_tasks:
                aug_features_1 = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_1 = model.pt_mlp(aug_features_1)
                aug_features_2 = model.pt_mlp(aug_features_2)
                logits_per_aug1 = aug_features_1 @ aug_features_2.t()/opt.nce_temp
                logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/opt.nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                loss = opt.lam0*(loss_1 + loss_2)/2
            if 'denoising' in opt.pt_tasks:
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                print(cat_outs.shape, con_outs.shape)
                # if con_outs.shape(-1) != 0:
                # import ipdb; ipdb.set_trace()
                if len(con_outs) > 0:
                    con_outs = torch.cat(con_outs,dim=1)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                l1 = 0
                # import ipdb; ipdb.set_trace()
                n_cat = x_categ.shape[-1]
                for j in range(1,n_cat):
                    l1+= criterion1(cat_outs[j],x_categ[:,j])
                loss += opt.lam2*l1 + opt.lam3*l2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        early_stopping(running_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练
        print(f'Epoch: {epoch}, Running Loss: {running_loss}')

    print('END OF PRETRAINING!')
    return model
        # if opt.active_log:
        #     wandb.log({'pt_epoch': epoch ,'pretrain_epoch_loss': running_loss
        #     })
