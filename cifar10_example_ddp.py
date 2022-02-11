'''Train CIFAR10 with AutoLRS in PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR 
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import os
import copy
import socket
import logging
import time
import pickle
from pathlib import Path
import numpy as np

from models import *
from autolrs.autolrs_callback import AutoLRS

logging.basicConfig(level=logging.DEBUG)

best_acc = 0 # best test accuracy
VAL_LEN = 10 # evaluate the validation loss on a small subset of the validation set which contains 10 mini-batches
AMP = True

dist.init_process_group(backend='nccl', init_method='env://')                                         
world_size = dist.get_world_size()
global_rank = dist.get_rank()
local_rank = int(os.environ.get('LOCAL_RANK', 0))

if torch.cuda.is_available():
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainsampler = torch.utils.data.DistributedSampler(
    trainset, num_replicas=world_size, rank=global_rank, shuffle=True
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, sampler=trainsampler, pin_memory=True, num_workers=2)
        

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testsampler = torch.utils.data.DistributedSampler(
    testset, num_replicas=world_size, rank=global_rank, shuffle=False
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, sampler=testsampler, pin_memory=True, num_workers=2)

valset = torch.utils.data.Subset(testset, range(VAL_LEN))
valsampler = torch.utils.data.DistributedSampler(
    valset, num_replicas=world_size, rank=global_rank, shuffle=False
)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=128, sampler=valsampler, pin_memory=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
net = VGG('VGG16')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()

start_epoch = 1

# restart
checkpoint = None
if Path('./checkpoint/best_ckpt.pth').exists(): 
    checkpoint = torch.load('./checkpoint/best_ckpt.pth',
                            map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch'] + 1
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']

net = net.to(device)
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
cudnn.benchmark = True
            
criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

# AMP
scaler = torch.cuda.amp.GradScaler(enabled=AMP)

def all_reduce(tensor, reduction='sum'):
    if dist.is_available():
        dist.all_reduce(tensor)
        if reduction == 'mean':
            tensor /= dist.get_world_size()
        dist.barrier()


def val_fn():
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            all_reduce(loss, reduction='mean')
            val_loss += loss.item()
    net.train()
    return val_loss

# scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1) # baseline LR schedule
scheduler = AutoLRS(net, optimizer, val_fn, min_lr=1e-3, max_lr=1e-1,
                    tau_ini=1000, tau_max=8000, tau_dash_ratio=0.1, k=10,)
if checkpoint is not None:
    scheduler.load_state_dict(checkpoint['scheduler'])

# Training
def train(start_epoch=1, max_epoch=350*2):
    for epoch in range(start_epoch, max_epoch+1,1): # multiply 2 to your original training epochs because the search steps of AutoLRS is the same as the actual training steps
        if global_rank==0:
            print('Epoch: %d' % (epoch))
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss = loss.detach().clone()
            all_reduce(loss, reduction='mean')
            scheduler.step(loss.item())

            if global_rank==0:
                print(f"Epoch: {epoch} - [{batch_idx+1}/{len(trainloader)}] - loss: {loss.item():.6f} - lr: {optimizer.param_groups[0]['lr']:.6f} - global_rank: {global_rank}")
                   
        test(epoch)

# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            
            all_reduce(loss, reduction='mean')
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        _tensor = torch.tensor([total, correct]).to(device)
        all_reduce(_tensor, reduction='sum')
        total, correct = _tensor.cpu().numpy()

    if global_rank==0:
        print("Test acc: ", correct/total)
    if correct/total > best_acc:
        best_acc = correct/total
        if local_rank==0:
            print('Saving..')
            state = {
                'epoch': epoch,
                'net': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'acc': best_acc
            }
            if not Path('./checkpoint').exists():
                Path('./checkpoint').mkdir(parents=True, exist_ok=True)
            torch.save(state, './checkpoint/best_ckpt.pth')
            print("Best acc: ", best_acc)
    net.train()
    
train(start_epoch)
if global_rank==0:
    print("Model saved to ./checkpoint/best_ckpt.pth")
