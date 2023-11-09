import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
from dl_assignment_7_common import *  # Your functions should go here if you want to use them from scripts
import os
import numpy as np
import torch.nn.utils.prune as prune

print('start')

run_name = 'iterative_random'

cp_path = f'./checkpoints/{run_name}'

if not os.path.exists(cp_path):
    os.makedirs(cp_path)

result_path = f'./results/{run_name}'
if not os.path.exists(result_path):
    os.makedirs(result_path)

device = d2l.try_gpu()
dataset_used = get_dataset('mnist', dir = './data', batch_size = 60, shuffle = True, download = False)

runs = 3
remaining_percentages = np.array([100, 15, 10, 7, 4, 3.5])
prune_fractions = (100-remaining_percentages)/100

epochs = [15,20, 20, 20, 25, 30]


early_stop_iterations = np.zeros([len(prune_fractions),runs])
early_stop_iterations = np.insert(early_stop_iterations, 0, prune_fractions, axis=1)

early_stop_trainacc = np.zeros([len(prune_fractions),runs])
early_stop_trainacc = np.insert(early_stop_trainacc, 0, prune_fractions, axis=1)

early_stop_testacc = np.zeros([len(prune_fractions),runs])
early_stop_testacc = np.insert(early_stop_testacc, 0, prune_fractions, axis=1)

n_rounds = 6

for j in range(runs):
# for j in range(1,2):    
    for i, fraction in enumerate(prune_fractions):
        print(f'run {j} for fraction {fraction}')
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        _, early_stop_values = train(net, optimizer, dataset_used, epochs = 10, file_specifier = f'LTH_fig4a_base', val_interval = 2, cp_path=cp_path , plot = False)
        
        for k in range(1,n_rounds):
            
            iter_frac = fraction**(1/n_rounds)
            if k == 1:
                trained_net = torch.load(f"{cp_path}/model_LeNet-after-LTH_fig4a_base.pth")
            else:
                trained_net = torch.load(f"{cp_path}/model_LeNet-after-LTH_4a_L1_pruned{fraction}.pth")
            
            net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
            net.load_state_dict(trained_net.state_dict())
            mask = L1_prune(net, iter_frac)
            
            # init_net = torch.load(f"{cp_path}/model_LeNet-before-LTH_fig4a_base.pth") # off for random reinit
            net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
            # net.load_state_dict(init_net.state_dict()) # off for random reinit
            init_net_pruned = prune_using_mask(net, mask)
            optimizer = torch.optim.Adam(init_net_pruned.parameters(), lr=0.0012)
            _, early_stop_values = train(init_net_pruned, optimizer, dataset_used, epochs = epochs[i], file_specifier = f'LTH_4a_L1_pruned{fraction}', val_interval = 2, cp_path=cp_path , plot = False)
        
        early_stop_iterations[i+1,j] = early_stop_values['iteration']
        early_stop_trainacc[i+1,j] = early_stop_values['train_acc']
        early_stop_testacc[i+1,j] = early_stop_values['test_acc']    
        
np.savetxt(f'{result_path}/early_stop_iterations_{run_name}.txt',early_stop_iterations)
np.savetxt(f'{result_path}/early_stop_trainacc_{run_name}.txt',early_stop_trainacc)
np.savetxt(f'{result_path}/early_stop_testacc_{run_name}.txt',early_stop_testacc)
