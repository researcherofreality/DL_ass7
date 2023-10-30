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

if not os.path.exists('results'):
    os.makedirs('results')

device = d2l.try_gpu()
dataset_used = get_dataset('mnist', dir = './data', batch_size = 60, shuffle = True, download = False)
L1_size, random_size = 5, 10

remaining_percentages = np.array([100, 70, 41.1, 16.9, 7.0, 2.9, 1.2, 0.5, 0.2])
prune_fractions = (100-remaining_percentages)/100

epochs_random = [10,10,16,20,40,50,50,50]
epochs_L1 = [10,10,10,10,10,20,50,50]

early_stop_iterations_lenet_L1 = np.zeros([len(prune_fractions),L1_size])
early_stop_iterations_lenet_random = np.zeros([len(prune_fractions),random_size])
                                       
early_stop_testacc_lenet_L1 = np.zeros([len(prune_fractions),L1_size])
early_stop_testacc_lenet_random = np.zeros([len(prune_fractions),random_size])

for j in range(L1_size):
    for i, fraction in enumerate(prune_fractions):
        print(f'run {j} with fraction {fraction}')
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        mask = L1_prune(net, fraction)
        net2, optimizer2 = create_network(arch = 'LeNet', input = 784, output = 10)
        net2 = prune_using_mask(net2,mask)
        _, early_stop_values = train(net2, optimizer2, dataset_used, epochs = epochs_L1[i], file_specifier = f'LTH_{j}', val_interval = 2, plot = False)
        early_stop_iterations_lenet_L1[i,j] = early_stop_values['iteration']
        early_stop_testacc_lenet_L1[i,j] = early_stop_values['test_acc']

early_stop_iterations_lenet_L1_avg = np.mean(early_stop_iterations_lenet_L1,axis=1)
early_stop_testacc_lenet_L1_avg = np.mean(early_stop_testacc_lenet_L1,axis=1)

np.savetxt('results/early_stop_iterations_lenet_L1.txt',early_stop_iterations_lenet_L1)
np.savetxt('results/early_stop_testacc_lenet_L1.txt',early_stop_testacc_lenet_L1)

print('start random')
for j in range(random_size):
    for i, fraction in enumerate(prune_fractions):
        print(f'run {j} with fraction {fraction}')
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        mask = random_prune(net, fraction)
        net2, optimizer2 = create_network(arch = 'LeNet', input = 784, output = 10)
        net2 = prune_using_mask(net2,mask)
        _, early_stop_values = train(net2, optimizer2, dataset_used, epochs = epochs_random[i], file_specifier = f'random_{j}', val_interval = 2, plot = False)
        early_stop_iterations_lenet_random[i,j] = early_stop_values['iteration']
        early_stop_testacc_lenet_random[i,j] = early_stop_values['test_acc']


early_stop_iterations_lenet_random_avg = np.mean(early_stop_iterations_lenet_random,axis=1)
early_stop_testacc_lenet_random_avg = np.mean(early_stop_testacc_lenet_random,axis=1)

np.savetxt('results/early_stop_iterations_lenet_random.txt',early_stop_iterations_lenet_random)
np.savetxt('results/early_stop_testacc_lenet_random.txt',early_stop_testacc_lenet_random)

plt.figure(figsize=(10,5))
plt.errorbar(remaining_percentages,early_stop_iterations_lenet_L1_avg,yerr=np.std(early_stop_iterations_lenet_L1,axis=1),label='L1')
plt.errorbar(remaining_percentages,early_stop_iterations_lenet_random_avg,yerr=np.std(early_stop_iterations_lenet_random,axis=1),label='random')
plt.gca().invert_xaxis()
plt.xlabel('Pruning fraction')
plt.ylabel('Number of iterations')
plt.legend()
plt.savefig('results/early_stop_iterations_lenet.png')
plt.show()

plt.figure(figsize=(10,5))
plt.errorbar(remaining_percentages,early_stop_testacc_lenet_L1_avg,yerr=np.std(early_stop_testacc_lenet_L1,axis=1),label='L1')
plt.errorbar(remaining_percentages,early_stop_testacc_lenet_random_avg,yerr=np.std(early_stop_testacc_lenet_random,axis=1),label='random')
plt.gca().invert_xaxis()
plt.xlabel('Pruning fraction')
plt.ylabel('Test accuracy')
plt.legend()
plt.savefig('results/early_stop_testacc_lenet.png')
plt.show()

