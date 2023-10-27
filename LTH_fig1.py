import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
from dl_assignment_7_common import *  # Your functions should go here if you want to use them from scripts
import os
import numpy as np
import torch.nn.utils.prune as prune

device = d2l.try_gpu()
dataset_used = get_dataset('mnist', dir = './data', batch_size = 60, shuffle = True, download = False)
fractions, L1_size, random_size = np.linspace(0,1,11), 3, 5

prune_fractions = np.linspace(0,1,fractions)
early_stop_iterations_lenet_L1 = np.zeros([fractions,L1_size])
early_stop_iterations_lenet_random = np.zeros([fractions,random_size])
                                       
early_stop_testacc_lenet_L1 = np.zeros([fractions,L1_size])
early_stop_testacc_lenet_random = np.zeros([fractions,random_size])

for j in range(L1_size):
    for i, fraction in enumerate(prune_fractions):
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        _ = L1_prune(net, fraction)
        _, early_stop_values = train(net, optimizer, dataset_used, epochs = 10, file_specifier = f'LTH_{j}', val_interval = 2, plot = False)
        early_stop_iterations_lenet_L1[i,j] = early_stop_values['iteration']
        early_stop_testacc_lenet_L1[i,j] = early_stop_values['test_acc']

for j in range(random_size):
    for i, fraction in enumerate(prune_fractions):
        net, optimizer = create_network(arch = 'LeNet', input = 784, output = 10)
        _ = random_prune(net, fraction)
        _, early_stop_values = train(net, optimizer, dataset_used, epochs = 10, file_specifier = f'random_{j}', val_interval = 2, plot = False)
        early_stop_iterations_lenet_random[i,j] = early_stop_values['iteration']
        early_stop_testacc_lenet_random[i,j] = early_stop_values['test_acc']

early_stop_iterations_lenet_L1_avg = np.mean(early_stop_iterations_lenet_L1,axis=1)
early_stop_iterations_lenet_random_avg = np.mean(early_stop_iterations_lenet_random,axis=1)

early_stop_testacc_lenet_L1_avg = np.mean(early_stop_testacc_lenet_L1,axis=1)
early_stop_testacc_lenet_random_avg = np.mean(early_stop_testacc_lenet_random,axis=1)

# save values to text file
np.savetxt('early_stop_iterations_lenet_L1.txt',early_stop_iterations_lenet_L1)
np.savetxt('early_stop_iterations_lenet_random.txt',early_stop_iterations_lenet_random)
np.savetxt('early_stop_testacc_lenet_L1.txt',early_stop_testacc_lenet_L1)
np.savetxt('early_stop_testacc_lenet_random.txt',early_stop_testacc_lenet_random)

# plot the results with errorbars and save it to an image
plt.figure(figsize=(10,5))
plt.errorbar(prune_fractions,early_stop_iterations_lenet_L1_avg,yerr=np.std(early_stop_iterations_lenet_L1,axis=1),label='L1')
plt.errorbar(prune_fractions,early_stop_iterations_lenet_random_avg,yerr=np.std(early_stop_iterations_lenet_random,axis=1),label='random')
plt.xlabel('Pruning fraction')
plt.ylabel('Number of iterations')
plt.legend()
plt.savefig('early_stop_iterations_lenet.png')
plt.show()

# plot the test accuracies with errorbars
plt.figure(figsize=(10,5))
plt.errorbar(prune_fractions,early_stop_testacc_lenet_L1_avg,yerr=np.std(early_stop_testacc_lenet_L1,axis=1),label='L1')
plt.errorbar(prune_fractions,early_stop_testacc_lenet_random_avg,yerr=np.std(early_stop_testacc_lenet_random,axis=1),label='random')
plt.xlabel('Pruning fraction')
plt.ylabel('Test accuracy')
plt.legend()
plt.savefig('early_stop_testacc_lenet.png')
plt.show()

