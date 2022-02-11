'''Train CIFAR10 with AutoLRS in PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

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

if torch.cuda.is_available():
    device = 'cuda'
    n_gpus = torch.cuda.device_count()
else:
    device = 'cpu'
    n_gpus = 1

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
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128*n_gpus, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100*n_gpus, shuffle=False, num_workers=2)

valset = torch.utils.data.Subset(testset, range(VAL_LEN))
valloader = torch.utils.data.DataLoader(
    valset, batch_size=128*n_gpus, shuffle=False, num_workers=2)

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

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net = net.to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

def val_fn():
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
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
        print('Epoch: %d' % (epoch))
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            print(f"Epoch: {epoch} - [{batch_idx+1}/{len(trainloader)}] - loss: {loss.item():.6f} - lr: {optimizer.param_groups[0]['lr']:.6f}")

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
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print("Test acc: ", correct/total)
    if correct/total > best_acc:
        best_acc = correct/total
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
print("Model saved to ./checkpoint/best_ckpt.pth")
