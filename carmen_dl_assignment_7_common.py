# This is a file where you should put your own functions
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import os
import torch.nn.utils.prune as prune
import copy 


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

def get_dataset(name, dir, batch_size=60, shuffle=True, download=False):
    """
    Returns a dictionary of PyTorch dataset objects train, val, and test for the specified dataset name.

    Args:
        - name (str): Name of the dataset. Currently supported options are 'fashionmnist' and 'mnist'.
        - dir (str): Directory to store the downloaded dataset.
        - batch_size (int, optional): Batch size for the data loader. Defaults to 60.
        - shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        - download (bool, optional): Whether to download the dataset if it is not already present. Defaults to False.

    Returns:
        - torch.utils.data.Dataset: Dictionary of PyTorch dataset objects train, val, and test for the specified dataset.
    """
    if not (os.path.exists(dir)):
        os.makedirs(dir)

    if name == 'fashionmnist':
        return get_fashionmnist(dir, batch_size, shuffle, download)
    elif name == 'mnist':
        return get_mnist(dir, batch_size, shuffle, download)
    elif name == 'cifar10':
        return get_cifar10(dir, batch_size, shuffle, download)
    else:
        raise ValueError(f'Unknown dataset: {name}')

def get_mnist(dir, batch_size, shuffle, download):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    mnist_dataset = torchvision.datasets.MNIST(root=dir, train=True, download=download, transform=transform)

    train_dataset, val_dataset = random_split(mnist_dataset, [55000, 5000])
    test_dataset = torchvision.datasets.MNIST(root=dir, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return {'train':train_loader,'val': val_loader, 'test': test_loader}

def get_fashionmnist(dir, batch_size, shuffle, download):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    fashionmnist_dataset = torchvision.datasets.FashionMNIST(root=dir, train=True, download=download, transform=transform)

    train_dataset, val_dataset = random_split(fashionmnist_dataset, [55000, 5000])
    test_dataset = torchvision.datasets.FashionMNIST(root=dir, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return {'train':train_loader,'val': val_loader, 'test': test_loader}

def get_cifar10(dir, batch_size, shuffle, download):    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    cifar10_dataset = torchvision.datasets.CIFAR10(root=dir, train=True, download=download, transform=transform)

    train_dataset, val_dataset = random_split(cifar10_dataset, [45000, 5000])
    test_dataset = torchvision.datasets.CIFAR10(root=dir, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return {'train':train_loader,'val': val_loader, 'test': test_loader}

# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------
def create_network(arch, lr=None, **kwargs):
    """
    Creates a neural network based on the specified architecture.

    Args:
        - arch (str): The name of the architecture to use. Currently supported options are 'LeNet' and 'arch2'.
        - lr (float, optional): The learning rate to use for the optimizer. If not provided, a default value will be used.
        - **kwargs: Additional keyword arguments to pass to the network constructor.

    Returns:
        - A tuple containing the created neural network and its optimizer.
    """
    
    if arch == 'LeNet':
        net = LeNet(**kwargs)
        if lr is None:
            lr = 0.0012
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        return net, optimizer
    
    elif arch == 'ResNet18':
        net = ResNet18(**kwargs)
        if lr is None:
            lr = 0.1
        if isinstance(lr, list):
            optimizer = []
            for l_r in lr:
                optimizer.append(torch.optim.SGD(net.parameters(), lr=l_r, momentum=0.9))
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        return net, optimizer
        # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        # return net, optimizer
    
    else:
        raise Exception(f"Unknown architecture: {arch}")

class LeNet(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.arch = 'LeNet'
        self.net = nn.Sequential(
            nn.Linear(input,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100,output)                
        )
    def forward(self,x):
        x = x.reshape(-1,784) # Flatten the image
        return self.net(x)
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = "ResNet18"
        self.net = models.resnet18()

    def forward(self, x):
        return self.net(x)
    
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = "VGG19"
        self.net = models.vgg19()

    def forward(self, x):
        return self.net(x)
    
# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

def train(net, optimizer, data, epochs, file_specifier = '', device = d2l.try_gpu(), val_interval = 5, cp_path = None, plot = False):
    """
    Trains a neural network model using the given optimizer and data.

    Parameters:
        - net (torch.nn.Module): The neural network model to train.
        - optimizer (torch.optim.Optimizer): The optimizer to use for training.
        - data (dict): A dictionary containing the training, validation, and test data.
        - epochs (int): The number of epochs to train the model for.
        - file_specifier (str): A string to append to the checkpoint file names.
        - device (torch.device): The device to use for training.
        - val_interval (int): The interval at which to perform validation.
        - cp_path (str): The path to save the model checkpoints.
        - plot (bool): Whether to plot the training and validation losses and accuracies.

    Returns:
        - tuple: A tuple containing two dictionaries, one for the model performance and one for the early stopping performance.
    """
    
    epochs += 1
    if (epochs - 1) % val_interval != 0:
        raise ValueError('epochs must be a multiple of val_interval')
    
    if cp_path is None:
        cp_path = './checkpoints'
        
    if not (os.path.exists(cp_path)):
        os.makedirs(cp_path)
    
    early_stopping = None
    best_val_loss = float('inf')
    
    net.to(device)
    loss = nn.CrossEntropyLoss()    
    
    torch.save(net, f'{cp_path}/model_{net.arch}-before-{file_specifier}.pth')
    
    if plot:
        animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs-1], figsize=(8, 5),
                                legend=['train loss', 'train accuracy', 'val loss', 'val accuracy'])

    iteration_count = 0
    for epoch in range(epochs):
        metric = {'train': d2l.Accumulator(3), 'val': d2l.Accumulator(3)}
        
        # training
        net.train()
        for i, (X, y) in enumerate(data['train']):
            iteration_count += 1
            X, y = X.to(device), y.to(device, torch.long)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            
            for name, param in net.named_parameters():
                if 'weight' in name:  
                    param.grad.data *= param.data.ne(0).float()
                    
            optimizer.step()
            metric['train'].add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            
        # validation & early stopping
        
        if epoch % val_interval == 0:
            net.eval()
            with torch.no_grad():
                for X_val, y_val in data['val']:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    y_val_hat = net(X_val)
                    l_val = loss(y_val_hat, y_val)
                    metric['val'].add(l_val * X_val.shape[0], d2l.accuracy(y_val_hat, y_val), X_val.shape[0])
                
                
            avg_val_loss = metric['val'][0] / metric['val'][2]
            if avg_val_loss < best_val_loss and epoch > 0:
                best_val_loss = avg_val_loss
                
                if os.path.exists(f'{cp_path}/model_{net.arch}-earlystop_iter_{early_stopping}-{file_specifier}.pth') and early_stopping is not None:
                    os.remove(f'{cp_path}/model_{net.arch}-earlystop_iter_{early_stopping}-{file_specifier}.pth')
                
                early_stopping = iteration_count
                early_stop_values = {'iteration': iteration_count, 'train_loss': metric['train'][0] / metric['train'][2], 'train_acc': metric['train'][1] / metric['train'][2], 'val_loss': avg_val_loss,'val_acc': metric['val'][1] / metric['val'][2],'test_acc': 0 }
                torch.save(net, f'{cp_path}/model_{net.arch}-earlystop_iter_{early_stopping}-{file_specifier}.pth')
                
            if plot:
                animator.add(epoch, (metric['train'][0] / metric['train'][2], metric['train'][1] / metric['train'][2], metric['val'][0] / metric['val'][2], metric['val'][1] / metric['val'][2]))
            
    
    torch.save(net, f'{cp_path}/model_{net.arch}-after-{file_specifier}.pth')

    train_loss, train_acc = metric['train'][0] / metric['train'][2], metric['train'][1] / metric['train'][2]
    val_loss, val_acc = metric['val'][0] / metric['val'][2], metric['val'][1] / metric['val'][2]
    test_acc = d2l.evaluate_accuracy_gpu(net, data['test'])
    model_performance = {'train loss': train_loss, 'train acc': train_acc, 
                         'val loss': val_loss,'val acc': val_acc, 'test acc': test_acc}
    print('model performance', model_performance)

    # check early stopping epoch
    early_net, _ = create_network(arch = 'LeNet', input = 784, output = 10)
    early_net_state = torch.load(f'{cp_path}/model_{net.arch}-earlystop_iter_{early_stopping}-{file_specifier}.pth')
    early_net.load_state_dict(early_net_state.state_dict())
    early_test_acc = d2l.evaluate_accuracy_gpu(early_net, data['test'])
    early_stop_values['test_acc'] = early_test_acc
    
    print('early stop model performance', early_stop_values)
    
    return model_performance, early_stop_values

def train_iterative(net, optimizer, data, epochs, file_specifier = '', device = d2l.try_gpu(), val_interval = 5, cp_path = None, plot = False):
    """
    Trains a neural network model using the given optimizer and data.

    Parameters:
        - net (torch.nn.Module): The neural network model to train.
        - optimizer (torch.optim.Optimizer): The optimizer to use for training.
        - data (dict): A dictionary containing the training, validation, and test data.
        - epochs (int): The number of epochs to train the model for.
        - file_specifier (str): A string to append to the checkpoint file names.
        - device (torch.device): The device to use for training.
        - val_interval (int): The interval at which to perform validation.
        - cp_path (str): The path to save the model checkpoints.
        - plot (bool): Whether to plot the training and validation losses and accuracies.

    Returns:
        - tuple: A tuple containing two dictionaries, one for the model performance and one for the early stopping performance.
    """
    
    epochs += 1
    if (epochs - 1) % val_interval != 0:
        raise ValueError('epochs must be a multiple of val_interval')
    
    if cp_path is None:
        cp_path = './checkpoints'
        
    if not (os.path.exists(cp_path)):
        os.makedirs(cp_path)
    
    early_stopping = None
    best_val_loss = float('inf')
    
    net.to(device)
    loss = nn.CrossEntropyLoss()    
    
    torch.save(net, f'{cp_path}/model_{net.arch}-before-{file_specifier}.pth')
    
    if plot:
        animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs-1], figsize=(8, 5),
                                legend=['train loss', 'train accuracy', 'val loss', 'val accuracy'])

    iteration_count = 0
    for epoch in range(epochs):
        metric = {'train': d2l.Accumulator(3), 'val': d2l.Accumulator(3)}
        
        # training
        net.train()
        for i, (X, y) in enumerate(data['train']):
            iteration_count += 1
            X, y = X.to(device), y.to(device, torch.long)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            
            for name, param in net.named_parameters():
                if 'weight' in name:  
                    param.grad.data *= param.data.ne(0).float()
                    
            optimizer.step()
            metric['train'].add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            
        # validation & early stopping
        
        if epoch % val_interval == 0:
            net.eval()
            with torch.no_grad():
                for X_val, y_val in data['val']:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    y_val_hat = net(X_val)
                    l_val = loss(y_val_hat, y_val)
                    metric['val'].add(l_val * X_val.shape[0], d2l.accuracy(y_val_hat, y_val), X_val.shape[0])
                
                
            avg_val_loss = metric['val'][0] / metric['val'][2]
            if avg_val_loss < best_val_loss and epoch > 0:
                best_val_loss = avg_val_loss
                
                if os.path.exists(f'{cp_path}/model_{net.arch}-earlystop_iter_{early_stopping}-{file_specifier}.pth') and early_stopping is not None:
                    os.remove(f'{cp_path}/model_{net.arch}-earlystop_iter_{early_stopping}-{file_specifier}.pth')
                
                early_stopping = iteration_count
                early_stop_values = {'iteration': iteration_count, 'train_loss': metric['train'][0] / metric['train'][2], 'train_acc': metric['train'][1] / metric['train'][2], 'val_loss': avg_val_loss,'val_acc': metric['val'][1] / metric['val'][2],'test_acc': 0 }
                torch.save(net, f'{cp_path}/model_{net.arch}-earlystop_iter_{early_stopping}-{file_specifier}.pth')
                
            if plot:
                animator.add(epoch, (metric['train'][0] / metric['train'][2], metric['train'][1] / metric['train'][2], metric['val'][0] / metric['val'][2], metric['val'][1] / metric['val'][2]))
            
    
    torch.save(net, f'{cp_path}/model_{net.arch}-after-{file_specifier}.pth')

    train_loss, train_acc = metric['train'][0] / metric['train'][2], metric['train'][1] / metric['train'][2]
    val_loss, val_acc = metric['val'][0] / metric['val'][2], metric['val'][1] / metric['val'][2]
    test_acc = d2l.evaluate_accuracy_gpu(net, data['test']) 
    model_performance = {'train loss': train_loss, 'train acc': train_acc, 
                         'val loss': val_loss,'val acc': val_acc, 'test acc': test_acc}
    print('model performance', model_performance)

    # check early stopping epoch
    early_net, _ = create_network(arch = 'LeNet', input = 784, output = 10)
    early_net_state = torch.load(f'{cp_path}/model_{net.arch}-earlystop_iter_{early_stopping}-{file_specifier}.pth')
    early_net.load_state_dict(early_net_state.state_dict())
    early_test_acc = d2l.evaluate_accuracy_gpu(early_net, data['test'])
    early_stop_values['test_acc'] = early_test_acc
    
    print('early stop model performance', early_stop_values)
    
    #return model_performance, early_stop_values
    return net

# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

def random_prune(net, fraction):
    """
    Randomly prunes the weights of the given neural network by the specified fraction.
    
    Args:
    - net: The neural network to be pruned.
    - fraction: The fraction of weights to be pruned.
    
    Returns:
    - mask: A list of masks for each layer of the neural network, where each mask specifies which weights have been pruned.
    """
    mask = []
    for i, layer in enumerate(net.net):
        
        if i == len(net.net):
            if isinstance(layer,nn.Linear):
                mask.append(prune.random_unstructured(layer, name='weight', amount=fraction/2))
            break
        
        if isinstance(layer,nn.Linear):
            mask.append(prune.random_unstructured(layer, name='weight', amount=fraction))
            
    return mask

def L1_prune(net, fraction, conv_fraction = 0):
    """
    Applies L1 unstructured pruning to the weights of the linear layers in the given network.

    Args:
        -  net (torch.nn.Module): The network to prune.
        - fraction (float): The fraction of weights to prune.

    Returns:
        - list: A list of masks for each pruned layer.
    """
    
    mask = []
    for i, layer in enumerate(net.net):
        
        if i == len(net.net):
            if isinstance(layer,nn.Linear):
                mask.append(prune.l1_unstructured(layer, name='weight', amount=fraction/2))
            elif isinstance(layer,torch.nn.Conv2d): # añadido
                mask.append(prune.l1_unstructured(layer, name='weight', amount=conv_fraction/2))
            elif isinstance(layer,models.resnet.BasicBlock): 
                mask.append(prune.l1_unstructured(layer, name='weight', amount=fraction/2))
            break
        
        if isinstance(layer,nn.Linear):
            mask.append(prune.l1_unstructured(layer, name='weight', amount=fraction))
        elif isinstance(layer,torch.nn.Conv2d): # añadido
            mask.append(prune.l1_unstructured(layer, name='weight', amount=conv_fraction))
        elif isinstance(layer,models.resnet.BasicBlock):
            mask.append(prune.l1_unstructured(layer, name='weight', amount=fraction)) #????
    return mask


def prune_using_mask(net, mask):
    """
    Prunes the network using the provided mask.

    Args:
        - net (nn.Module): The neural network to be pruned.
        - mask (list): A list of masks for each layer in the network.

    Returns:
        - nn.Module: The pruned neural network.
    """
    i = 0
    for j, layer in enumerate(net.net):
        if isinstance(layer, nn.Linear):
            prune.custom_from_mask(layer, 'weight', mask[i].weight_mask)
            i += 1
        elif isinstance(layer, nn.Conv2d):  # añadido
            prune.custom_from_mask(layer, 'weight', mask[i].weight_mask)
            i += 1
        elif isinstance(layer, models.resnet.BasicBlock):
            prune.custom_from_mask(layer, 'weight', mask[i].weight_mask)
            i += 1

    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight') 
        elif isinstance(module, nn.Conv2d): #estas añadiendo esto
            prune.remove(module, 'weight')
        elif isinstance(module, models.resnet.BasicBlock):
            prune.remove(module, 'weight')

    return net

####
def iterative_pruning(arch,mode, dataset, fraction, n_rounds, epochs, file_specifier = '', device = d2l.try_gpu(), val_interval = 5, cp_path = None, plot = False):
    """
    Prunes the network using iterative pruning. 

    Args:
        - net (nn.Module): The neural network to be pruned.
        - fraction (float): The fraction of weights to prune.

    Returns:
        - nn.Module: The pruned neural network.
    """
    mask_aux = []
    net_aux = []
    dataset_used = get_dataset(dataset,'./data')
    net, optimizer = create_network(arch = arch, input = 784, output = 10)
    if mode == 'Winning_ticket':
        for i in range(n_rounds):
            aux = L1_prune( net, fraction ** (1/(i+1)) )
            # problem with the mask: is it just prunning the already prunned weights?
            #print(aux[0].weight_mask)
            mask_aux.append(copy.deepcopy(aux[0].weight_mask))
            net = prune_using_mask(net, aux)
            # Training and obtaining 
            net = copy.deepcopy(train_iterative(net, optimizer, dataset_used, epochs, file_specifier , device , val_interval , cp_path , plot ) )
            net_aux.append(copy.deepcopy(net))
    elif mode == 'Random_reinit':
        # complete this 
        # Add that the 0 masked weights of the net are reinitialized to random
        print('work on it!!!!!')

        for i in range(n_rounds):
            aux = L1_prune( net, fraction ** (1/n_rounds) )
            #print(aux[0].weight_mask)
            mask_aux.append(copy.deepcopy(aux))
            net = prune_using_mask(net, aux)
            # Training and obtaining 
            net = train_iterative(net, optimizer, dataset_used, epochs, file_specifier , device , val_interval , cp_path , plot ) 
            net_aux.append(copy.deepcopy(net))
            # WHICH NUMBER SHOULD I USE?????
            # r_r = (torch.where((aux==0)|(aux==1), aux^1, aux) * torch.random)
            # aux = aux + r_r
    else: 
        raise Exception(f"Unknown architecture: {mode}")
    return mask_aux, net_aux



# -----------------------------------------------------------------------------
# Other
# -----------------------------------------------------------------------------

def winning_tickets(arch, dataset, epochs, val_interval, pruning_fraction, pruning_method = '', specifier = 'LTH_1'):
    """the loop defined in the Identifying Winning Tickets section of the paper"""
    dataset_used = get_dataset(dataset,'./data')

    net, optimizer = create_network(arch = arch, input = 784, output = 10)
    model_performance, early_stop_values = train(net, optimizer, dataset_used, epochs = epochs, file_specifier = specifier, val_interval = val_interval, plot = False)

    net, optimizer = create_network(arch = arch, input = 784, output = 10)
    retrain_net = torch.load(f'./checkpoints/model_{arch}-after-{specifier}.pth')
    net.load_state_dict(retrain_net.state_dict())
    
    if pruning_method == 'L1':
        mask = L1_prune(net, pruning_fraction)
    if pruning_method == 'random':
        mask = random_prune(net, pruning_fraction)

    net, optimizer = create_network(arch = arch, input = 784, output = 10)
    retrain_net = torch.load(f'./checkpoints/model_{arch}-before-{specifier}.pth')
    net.load_state_dict(retrain_net.state_dict())

    winning_ticket_net = prune_using_mask(net, mask)

    winning_ticket_net.eval()  # Set the network to evaluation mode
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X, y in dataset_used['test']:
            X, y = X.to(d2l.try_gpu()), y.to(d2l.try_gpu())
            y_hat = winning_ticket_net(X)
            l = nn.CrossEntropyLoss(y_hat, y)
            total_loss += l.item() * len(y)  # Accumulate batch loss
            total_samples += len(y)  # Accumulate batch size

    test_loss = total_loss / total_samples
    test_acc = d2l.evaluate_accuracy_gpu(winning_ticket_net, dataset_used['test'])
    
    return test_loss, test_acc
