# Packages
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import time

def gain(data_x, gain_parameters):
    '''Impute missing values in data_x
    Args:
        - data_x: original data with missing values
        - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
    
    Returns:
        - imputed_data: imputed data
    '''
    data_m = 1-np.isnan(data_x)
    print(data_m)
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    # use_gpu = gain_parameters['use_gpu']

    no, dim = data_x.shape
    h_dim = int(dim)

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    ## GAIN architecture
    cur_device = torch.cuda.current_device()
    #  Discriminator variables
    D_W1 = torch.tensor(xavier_init([dim*2, h_dim]),requires_grad=True, device=cur_device)     # Data + Hint as inputs
    D_b1 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True, device=cur_device)

    D_W2 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True, device=cur_device)
    D_b2 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True, device=cur_device)

    D_W3 = torch.tensor(xavier_init([h_dim, dim]),requires_grad=True, device=cur_device)
    D_b3 = torch.tensor(np.zeros(shape = [dim]),requires_grad=True, device=cur_device)       # Output is multi-variate
    
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    G_W1 = torch.tensor(xavier_init([dim*2, h_dim]),requires_grad=True, device=cur_device)     # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True, device=cur_device)

    G_W2 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True, device=cur_device)
    G_b2 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True, device=cur_device)

    G_W3 = torch.tensor(xavier_init([h_dim, dim]),requires_grad=True, device=cur_device)
    G_b3 = torch.tensor(np.zeros(shape = [dim]),requires_grad=True, device=cur_device)

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    optimizer_D = torch.optim.Adam(params=theta_D)
    optimizer_G = torch.optim.Adam(params=theta_G)

    ## GAIN functions
    # Generator
    def generator(new_x,m):
        inputs = torch.cat(dim = 1, tensors = [new_x,m])  # Mask + Data Concatenate
        G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)   
        G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output
        
        return G_prob
    
    # Discriminator
    def discriminator(new_x, h):
        inputs = torch.cat(dim = 1, tensors = [new_x,h])  # Hint + Data Concatenate
        D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
        D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
        D_logit = torch.matmul(D_h2, D_W3) + D_b3
        D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
        
        return D_prob

    
    def discriminator_loss(X, M, H):
        G_sample = generator(X, M)
        # Combine with observed data
        Hat_X = X * M + G_sample * (1-M)

        # Discriminator
        D_prob = discriminator(Hat_X, H)

        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
        return D_loss

    
    def generator_loss(X, M, H):
        G_sample = generator(X, M)
        Hat_X = X * M + G_sample * (1-M)
        D_prob = discriminator(Hat_X, H)

        G_loss_tmp = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
        MSE_loss = torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M.float())
        G_loss = G_loss_tmp + alpha * MSE_loss
        return G_loss, MSE_loss
         
    
    ## Iterations
    for it in tqdm(range(iterations)):
        # sample batch
        batch_idx = sample_idx(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        # Sample random vectors  
        Z_mb = sample_Z(0, 0.01, batch_size, dim)
        # Sample hint vectors
        H_mb_tmp = sample_M(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_tmp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        X_mb = torch.tensor(X_mb, device=cur_device)
        M_mb = torch.tensor(M_mb, device=cur_device)
        H_mb = torch.tensor(H_mb, device=cur_device)

        optimizer_D.zero_grad()
        D_loss_curr = discriminator_loss(X=X_mb, M=M_mb, H=H_mb)
        D_loss_curr.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        G_loss_curr, MSE_loss_curr = generator_loss(X=X_mb, M=M_mb, H=H_mb)
        G_loss_curr.backward()
        optimizer_G.step()

        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('MSE_loss: {:.4}'.format(np.sqrt(MSE_loss_curr.item())))
            print()

    
    ## Return imputed data
    Z_mb = sample_Z(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    # print(X_mb)
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

    X_mb = torch.tensor(X_mb, device=cur_device)
    M_mb = torch.tensor(M_mb, device=cur_device)

    imputed_data = generator(X_mb, M_mb)
    imputed_data = torch.tensor(data_m * norm_data_x, device=cur_device) + (1-M_mb) * imputed_data
    imputed_data = imputed_data.cpu().detach().numpy()
    # print(imputed_data)

    # renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)


def normalization(data, parameters=None):
    '''Normalize data in range [0,1]'''
    _, dim = data.shape
    norm_data = data.copy()
    if parameters is None:
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        
        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
        
        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                        'max_val': max_val}
    
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
    # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
        
        norm_parameters = parameters    

    return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.'''

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
    return renorm_data


def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.'''
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
      
    return rounded_data


def sample_idx(total, batch_size):
    '''Sample index of the mini_batch'''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def sample_M(p, rows, cols):
    '''Sample binary random variables.'''
    unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix


def sample_Z(low, high, rows, cols):
    '''Sample uniform random variables'''
    return np.random.uniform(low, high, size = [rows, cols]) 

def gain_main(data):
    gain_parameters = {
        'batch_size':128,
        'iterations': 10000,
        'hint_rate': 0.9,
        'alpha': 10,
    }  
    data_wo_labels = data
    data_wo_labels = np.array(data_wo_labels)
    # data = np.loadtxt(data_name, delimiter=",", skiprows=1, dtype='float32')
    print(data_wo_labels.shape)
    imputed_data = gain(data_x=data_wo_labels, gain_parameters=gain_parameters)
    imputed_data = pd.DataFrame(imputed_data)
    data = imputed_data
    return data

if __name__ == "__main__":
    miss_rate = 0.5
    dataset_name = "Gesture"
    time_start = time.time()  # 开始计时
    print(time_start)
    data_name = "data/Gesture/mcar_" + dataset_name + "_0.5.csv"
    data = pd.read_csv(data_name)
    data = gain_main(data)
    data.to_csv("data/" + dataset_name + "/mcar_" + dataset_name + "_" + str(miss_rate) + "_GAIN" + ".csv", index=None)
    print(data)
    time_end = time.time()  # 结束计时

    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')