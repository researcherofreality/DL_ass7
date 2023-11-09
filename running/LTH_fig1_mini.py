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

run_name = 'mini'
cp_path = f'./checkpoints/{run_name}'

if not os.path.exists(cp_path):
    os.makedirs(cp_path)

result_path = f'./results/{run_name}'
if not os.path.exists(result_path):
    os.makedirs(result_path)

device = d2l.try_gpu()
dataset_used = get_dataset('mnist', dir = './data', batch_size = 60, shuffle = True, download = False)

L1_size, random_size = 5, 10
remaining_percentages = np.array([ 2, 1, 0.7, 0.5, 0.2])

prune_fractions = (100-remaining_percentages)/100

epochs_random = [ 50, 50, 50, 50, 50]

epochs_L1 = [ 50, 50, 50, 50, 50]

early_stop_iterations_lenet_L1 = np.zeros([len(prune_fractions),L1_size])
early_stop_iterations_lenet_L1 = np.insert(early_stop_iterations_lenet_L1, 0, prune_fractions, axis=1)

early_stop_iterations_lenet_random = np.zeros([len(prune_fractions),random_size])
early_stop_iterations_lenet_random = np.insert(early_stop_iterations_lenet_random, 0, prune_fractions, axis=1)
                                       
early_stop_testacc_lenet_L1 = np.zeros([len(prune_fractions),L1_size])
early_stop_testacc_lenet_L1 = np.insert(early_stop_testacc_lenet_L1, 0, prune_fractions, axis=1)

early_stop_testacc_lenet_random = np.zeros([len(prune_fractions),random_size])
early_stop_testacc_lenet_random = np.insert(early_stop_testacc_lenet_random, 0, prune_fractions, axis=1)


# net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
# _, early_stop_values = train(net, optimizer, dataset_used, epochs = 50, file_specifier = f'LTH_fig1_base', val_interval = 2, plot = False)

for j in range(L1_size):
    for i, fraction in enumerate(prune_fractions):
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        _, early_stop_values = train(net, optimizer, dataset_used, epochs = 10, file_specifier = f'LTH_fig1_base', val_interval = 2, cp_path=cp_path , plot = False)
        
        print(f'run {j} for fraction {fraction} L1 pruned')
        trained_net = torch.load(f"{cp_path}/model_LeNet-after-LTH_fig1_base.pth")
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        net.load_state_dict(trained_net.state_dict())
        mask = L1_prune(net, fraction)
        
        init_net = torch.load(f"{cp_path}/model_LeNet-before-LTH_fig1_base.pth")
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        net.load_state_dict(init_net.state_dict())
        init_net_pruned = prune_using_mask(net, mask)
        optimizer = torch.optim.Adam(init_net_pruned.parameters(), lr=0.0012)
        
        _, early_stop_values = train(init_net_pruned, optimizer, dataset_used, epochs = epochs_L1[i], file_specifier = f'LTH_L1_pruned{fraction}', val_interval = 2, cp_path=cp_path , plot = False)
        
        early_stop_iterations_lenet_L1[i+1,j] = early_stop_values['iteration']
        early_stop_testacc_lenet_L1[i+1,j] = early_stop_values['test_acc']    

early_stop_iterations_lenet_L1_avg = np.mean(early_stop_iterations_lenet_L1[:,1:],axis=1)
early_stop_testacc_lenet_L1_avg = np.mean(early_stop_testacc_lenet_L1[:,1:],axis=1)

np.savetxt(f'{result_path}/early_stop_iterations_lenet_L1_{run_name}.txt',early_stop_iterations_lenet_L1)
np.savetxt(f'{result_path}/early_stop_testacc_lenet_L1_{run_name}.txt',early_stop_testacc_lenet_L1)

print('start random')
for j in range(random_size):
    for i, fraction in enumerate(prune_fractions):
        print(f'run {j} for fraction {fraction} random pruned')
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        _, early_stop_values = train(net, optimizer, dataset_used, epochs = 10, file_specifier = f'LTH_fig1_base', val_interval = 2, cp_path=cp_path , plot = False)
        
        trained_net = torch.load(f"{cp_path}/model_LeNet-after-LTH_fig1_base.pth")
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        net.state_dict(trained_net.state_dict())
        mask = random_prune(net, fraction)
        
        init_net = torch.load(f"{cp_path}/model_LeNet-before-LTH_fig1_base.pth")
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        net.state_dict(init_net.state_dict())
        init_net_pruned = prune_using_mask(net, mask)
        optimizer = torch.optim.Adam(init_net_pruned.parameters(), lr=0.0012)
        
        _, early_stop_values = train(init_net_pruned, optimizer, dataset_used, epochs = epochs_L1[i], file_specifier = f'LTH_random_pruned{fraction}', val_interval = 2, cp_path=cp_path , plot = False)
        
        early_stop_iterations_lenet_random[i+1,j] = early_stop_values['iteration']
        early_stop_testacc_lenet_random[i+1,j] = early_stop_values['test_acc']    

early_stop_iterations_lenet_random_avg = np.mean(early_stop_iterations_lenet_random[:,1:],axis=1)
early_stop_testacc_lenet_random_avg = np.mean(early_stop_testacc_lenet_random[:,1:],axis=1)

np.savetxt(f'{result_path}/early_stop_iterations_lenet_random_{run_name}.txt',early_stop_iterations_lenet_random)
np.savetxt(f'{result_path}/early_stop_testacc_lenet_random_{run_name}.txt',early_stop_testacc_lenet_random)

# # Equally spaced x values for plotting
# x_spaced = np.arange(len(prune_fractions))

# plt.figure(figsize=(10, 6))
# plt.errorbar(x_spaced,early_stop_iterations_lenet_random_avg,yerr=np.std(early_stop_iterations_lenet_random[:,1:],axis=1),label='random')
# plt.errorbar(x_spaced,early_stop_iterations_lenet_L1_avg,yerr=np.std(early_stop_iterations_lenet_L1[:,1:],axis=1),label='L1')
# plt.xticks(x_spaced, prune_fractions)
# plt.xlabel('Percent of Weights Remaining')
# plt.ylabel('EarlyStop Iteration (Val)')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('./results/fig1.png')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.errorbar(x_spaced,early_stop_testacc_lenet_random_avg,yerr=np.std(early_stop_testacc_lenet_random[:,1:],axis=1),label='random')
# plt.errorbar(x_spaced,early_stop_testacc_lenet_L1_avg,yerr=np.std(early_stop_testacc_lenet_L1[:,1:],axis=1),label='L1')
# plt.xticks(x_spaced, prune_fractions)
# plt.xlabel('Percent of Weights Remaining')
# plt.ylabel('Accuracy at Early-Stop (Test)')
# plt.legend()
# plt.grid(True)


# plt.tight_layout()
# plt.savefig('./results/fig2.png')
# plt.show()