import argparse
import copy
import numpy as np
import os
import pickle

import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, models, transforms

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import cifar_models

model_choices = [m for m in dir(models) if not m.startswith('__')]

parser = argparse.ArgumentParser(description='ImageNet Resnet Loss Landscape')
parser.add_argument('data', type=str, help="Path to data director")
parser.add_argument('model1', type=str, help="Model1 checkpoint")
parser.add_argument('-m2', '--model2', type=str, help="Model2 checkpoint")
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--viz-samples', type=int, default=100, help="# of interpolants to sample")
parser.add_argument('--cifar', action='store_true', help="Use cifar10 data and models, otherwise use Imagenet")
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--arch', type=str, default='resnet18', choices=model_choices, help="ResNet architecture")
parser.add_argument('--test-samples', type=int, default=50000, help="# testing samples to evaluate")
parser.add_argument('--disable-cuda', action='store_true')

args = parser.parse_args()

torch.manual_seed(0)
if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device("cpu")


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


def perturb(model, vector, eps):
    ret = copy.deepcopy(model)

    for param, v_i in zip(ret.parameters(), vector):
        param.requires_grad = False
        update = torch.from_numpy(eps * v_i).to(device)
        param += update.float() 

    return ret


def visualize_single(model, vector, testloader, trainloader, viz_samples, test_samples):
    if not vector:
        vector = []
        for p in model.parameters():
            vector.append(np.random.normal(size=p.shape))
        print('\tSaving perturbation_vector')
        with open('perturbation_vector', 'wb') as f:
            pickle.dump(vector, f)

    test_losses = []
    test_acces = []

    train_losses = []
    train_acces = []

    epses = np.linspace(left, right, num=viz_samples)

    for i, eps in enumerate(epses):
        print("\tTesting perturbation {}/{}".format(i+1, viz_samples))
        pert_model = perturb(model, vector, eps)
        test_loss, test_acc = test(testloader, pert_model, criterion, test_samples)
        train_loss, train_acc = test(trainloader, pert_model, criterion, test_samples)

        test_losses.append(test_loss)
        test_acces.append(test_acc)

        train_losses.append(train_loss)
        train_acces.append(train_acc)

    with open(os.path.join(args.output_dir, "raw_arrays"), "wb") as f:
        d = {
                "test loss": test_losses,
                "test accuracy": test_acces,
                "train loss": train_losses,
                "train accuracy": train_acces,
            }
        pickle.dump(d, f)

    plt.plot(epses, train_losses, label="Train Loss")
    plt.plot(epses, test_losses, label="Test Loss")
    plt.xlabel(u"\u03B5 (\u03B8' = \u03B8 + \u03B5)")
    plt.ylabel("Cross Entropy Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'loss.pdf'), dpi=150)

    plt.clf()

    plt.plot(epses, train_acces, label="Train Acc")
    plt.plot(epses, test_acces, label="Test Acc")
    plt.xlabel(u"\u03B5 (\u03B8' = \u03B8 + \u03B5)")
    plt.ylabel("Top 1 Accuracy %")
    plt.axis([left, right, 0, 100]) 
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'acc.pdf'), dpi=150)


def visualize(model1, model2, testloader, trainloader, viz_samples, test_samples):
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

        print("Test loss: {:.5f}\tTest acc: {:.3f}\tTrain loss: {:.5f}\tTrain acc: {:.3f}"\
                .format(test_loss, test_acc, train_loss, train_acc))
        
        test_losses.append(test_loss)
        test_acces.append(test_acc)

        train_losses.append(train_loss)
        train_acces.append(train_acc)


    with open(os.path.join(args.output_dir, "raw_arrays"), "wb") as f:
        d = {
                "test loss": test_losses,
                "test accuracy": test_acces,
                "train loss": train_losses,
                "train accuracy": train_acces,
            }
        pickle.dump(d, f)

    plt.plot(alphas, test_losses, label="Test Loss")
    plt.plot(alphas, train_losses, label="Train Loss")
    plt.xlabel(u"\u03B1 (\u03B8' = \u03B1 * w1 + (1 - \u03B1) * w2)")
    plt.ylabel("Cross Entropy Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "loss.pdf"), dpi=150)

    plt.clf()

    plt.plot(alphas, test_acces, label="Test Acc")
    plt.plot(alphas, train_acces, label="Train Acc")
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



if __name__ == '__main__':
    if not os.path.isdir(args.output_dir):
        os.makedirs(path)

    print('\nLoading data')

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if args.cifar:
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
    else:
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


    trainset = datasets.ImageFolder(traindir, transform_train)
    testset = datasets.ImageFolder(valdir, transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=30)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=30)

    print('Loading models')

    if args.cifar:
        model = getattr(cifar_models, args.arch)
    else:
        model = getattr(models, args.arch)

    model1 = model()
    checkpoint1 = torch.load(args.model1)
    model1.load_state_dict(checkpoint1['state_dict'])

    if args.model_2 is not None:
        model2 = model()
        checkpoint2 = torch.load(args.model2)
        model2.load_state_dict(checkpoint2['state_dict'])

        visualize(model1, model2, testloader, trainloader, args.viz_samples, args.test_samples)

    else:
        if args.vector:
            with open(args.vector, 'rb') as f:
                vector = pickle.load(f)
        else:
            vector = None
            
        visualize(model1, vector, testloader, trainloader, args.viz_samples, args.test_samples)

