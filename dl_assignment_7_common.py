# This is a file where you should put your own functions
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch.utils.data import DataLoader, random_split
import os
import torch.nn.utils.prune as prune

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

def get_dataset(name, dir, batch_size = 60, shuffle = True, download = False):
    if not (os.path.exists(dir)):
        os.makedirs(dir)
    
    if name == 'fashionmnist':
        return get_fashionmnist(dir, batch_size, shuffle, download)
    elif name == 'mnist':
        return get_mnist(dir, batch_size, shuffle, download)
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


# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------
def create_network(arch, lr = None, **kwargs):
    
    if arch == 'LeNet':
        net = LeNet(**kwargs)
        if lr is None:
            lr = 0.0012
        optimizer = torch.optim.Adam(net.parameters(), lr = lr )
        return net, optimizer
    
    elif arch == 'arch2':
        return create_network_arch2(**kwargs)
    
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



# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

def train(net, optimizer, data, epochs, file_specifier = '', device = d2l.try_gpu(), val_interval = 5, cp_path = None, plot = False):
    
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
            if iteration_count % 10000 == 0:
                print('10k iters')
            X, y = X.to(device), y.to(device, torch.long)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
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


# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

def L1_prune(net,fraction):
    mask = []
    for i, layer in enumerate(net.net):
        
        if i == len(net.net):
            if isinstance(layer,nn.Linear):
                mask.append(prune.l1_unstructured(layer, name='weight', amount=0.1))
            break
        
        if isinstance(layer,nn.Linear):
            mask.append(prune.l1_unstructured(layer, name='weight', amount=0.2))

    return mask

def random_prune(net,fraction):
    mask = []
    for i, layer in enumerate(net.net):
        
        if i == len(net.net):
            if isinstance(layer,nn.Linear):
                mask.append(prune.random_unstructured(layer, name='weight', amount=0.1))
            break
        
        if isinstance(layer,nn.Linear):
            mask.append(prune.random_unstructured(layer, name='weight', amount=0.2))
            
    return mask

def prune_using_mask(net,mask):
    i = 0
    for j, layer in enumerate(net.net):
        if isinstance(layer, nn.Linear):
            prune.custom_from_mask(layer, 'weight', mask[i].weight_mask)
            i += 1

    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight') 
    return net

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
