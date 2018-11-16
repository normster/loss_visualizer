import argparse
import copy
import numpy as np
import os
import pickle

import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, transforms

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import resnet

parser = argparse.ArgumentParser(description='ImageNet Resnet Loss Landscape')
parser.add_argument('model1', type=str, help="Model1 checkpoint")
parser.add_argument('model2', type=str, help="Model2 checkpoint")
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--viz-samples', type=int, default=100, help="# of interpolants to sample")
parser.add_argument('--data-dir', type=str, default='/rscratch/imagenet12_data')
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help="ResNet architecture")
parser.add_argument('--test-samples', type=int, default=50000, help="# testing samples to evaluate")
parser.add_argument('--disable-cuda', action='store_true')

args = parser.parse_args()

torch.manual_seed(0)
if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device("cpu")

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res


def interpolate(model1, model2, alpha):
    ret = copy.deepcopy(model1)
    
    ret_dict = ret.state_dict()
    dict1 = model1.state_dict()
    dict2 = model2.state_dict()

    for k in ret_dict:
        ret_dict[k] = alpha * dict1[k] + (1 - alpha) * dict2[k]

    ret.load_state_dict(ret_dict)
    return ret


def visualize(model1, model2, testloader, trainloader, trainloader2, viz_samples, test_samples):
    test_losses = []
    test_acces = []

    train_losses = []
    train_acces = []

    train2_losses = []
    train2_acces = []

    alphas = np.linspace(-1, 2, num=viz_samples)

    criterion = nn.CrossEntropyLoss()
    for i, alpha in enumerate(alphas):
        print("\nTesting perturbation {}/{}".format(i+1, viz_samples))
        interpolant = interpolate(model1, model2, alpha).to(device)

        test_loss, test_acc = test(testloader, interpolant, criterion, test_samples)
        train_loss, train_acc = test(trainloader, interpolant, criterion, test_samples)
        train2_loss, train2_acc = test(trainloader2, interpolant, criterion, test_samples)

        print("Test loss: {:.5f}\tTest acc: {:.3f}\tTrain loss: {:.5f}\tTrain acc: {:.3f}\tTrain2 loss: {:.5f}\tTrain2 acc: {:.3f}"\
                .format(test_loss, test_acc, train_loss, train_acc, train2_loss, train2_acc))
        
        test_losses.append(test_loss)
        test_acces.append(test_acc)

        train_losses.append(train_loss)
        train_acces.append(train_acc)

        train2_losses.append(train2_loss)
        train2_acces.append(train2_acc)

    with open(os.path.join(args.output_dir, "raw_arrays"), "wb") as f:
        d = {
                "test loss": test_losses,
                "test accuracy": test_acces,
                "train loss": train_losses,
                "train accuracy": train_acces,
                "train2 loss": train2_losses,
                "train2 accuracy": train2_acces,
            }
        pickle.dump(d, f)

    plt.plot(alphas, test_losses, label="Test Loss")
    plt.plot(alphas, train_losses, label="Train Loss")
    plt.plot(alphas, train2_losses, label="Train2 Loss")
    plt.xlabel(u"\u03B1 (\u03B8' = \u03B1 * w1 + (1 - \u03B1) * w2)")
    plt.ylabel("Cross Entropy Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "loss.pdf"), dpi=150)

    plt.clf()

    plt.plot(alphas, test_acces, label="Test Acc")
    plt.plot(alphas, train_acces, label="Train Acc")
    plt.plot(alphas, train2_acces, label="Train2 Acc")
    plt.xlabel(u"\u03B1 (\u03B8' = \u03B1 * w1 + (1 - \u03B1) * w2)")
    plt.ylabel("Top 1 Accuracy %")
    plt.axis([-1, 2, 0, 100]) 
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "acc.pdf"), dpi=150)


def test(loader, model, criterion, samples):
    model.eval()
   
    total_loss = 0.
    total_acc = 0.
    total = 0

    display_freq = samples // 5

    for inputs, targets in loader:
        if total >= samples:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)[0]

        total_loss += loss.item()
        total_acc += acc.item()
        total += targets.size(0)

        if total % display_freq == 0:
            print("\tLoss: {:.5f}\tAcc: {:.3f}".format(total_loss / total, 100 * total_acc / total))

    return total_loss / total, 100 * total_acc / total 


if not os.path.isdir(args.output_dir):
    mkdir_p(args.output_dir)

print('\nLoading data')

traindir = os.path.join(args.data_dir, 'train')
valdir = os.path.join(args.data_dir, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
transform_train=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.1,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

transform_train2 =transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

trainset = datasets.ImageFolder(traindir, transform_train)
trainset2 = datasets.ImageFolder(traindir, transform_train2)
testset = datasets.ImageFolder(valdir, transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=30)
trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=args.batch_size, shuffle=True, num_workers=30)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=30)

print('Loading models')

model = getattr(resnet, args.arch)
model1 = model()
model2 = model()

checkpoint1 = torch.load(args.model1)
checkpoint2 = torch.load(args.model2)

model1.load_state_dict(checkpoint1['state_dict'])
model2.load_state_dict(checkpoint2['state_dict'])

visualize(model1, model2, testloader, trainloader, trainloader2, args.viz_samples, args.test_samples)

