import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.backends.cudnn as cudnn

from torchvision.datasets import CIFAR10, ImageFolder, MNIST, FashionMNIST
import torchvision.transforms as transforms

import argparse
import logging
import os
import numpy as np

from sklearn import linear_model, neural_network
from sklearn.metrics import f1_score, accuracy_score, classification_report
from resnet9 import ResNet9, CNN
from dataset import ImageFolderDataset
from ema import ExponentialMovingAverage


def cnn_classify(train_set, test_set, fp, num_classes=10, logger=None, reverse_label=False, lr=3e-4):
    batch_size = 64
    max_epoch = 50
    criterion = nn.CrossEntropyLoss()
    #model = ResNet9(num_classes, 3)
    model = CNN(img_dim=(args.c, args.image_size, args.image_size), num_classes=num_classes).cuda()
    ema = ExponentialMovingAverage(model.parameters(), 0.9999)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=1)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, drop_last=False, num_workers=1)

    model.train()
    for epoch in range(max_epoch):
        total = 0
        correct = 0
        for _, (inputs, targets) in enumerate(train_loader):
            if len(targets.shape) == 2:
                inputs = (inputs.cuda().to(torch.float32) / 127.5 - 1.)
                targets = torch.argmax(targets, dim=1)
            if reverse_label:
                targets = 1 - targets
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            ema.update(model.parameters())
        train_acc = correct / total * 100
        #scheduler.step()
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        total = 0
        correct = 0
        y_list = []
        pred_list = []
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(test_loader):
                if len(targets.shape) == 2:
                    inputs = (inputs.cuda().to(torch.float32) / 127.5 - 1.)
                    targets = torch.argmax(targets, dim=1)
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                y_list.append(targets.detach().cpu())
                pred_list.append(predicted.detach().cpu())
        
        y_list = torch.cat(y_list).numpy()
        pred_list = torch.cat(pred_list).numpy()
        report = classification_report(y_list, pred_list)
        logger.info(report)
        
        test_acc = correct / total * 100

        logger.info("Epoch: {} Train acc: {} Test acc: {}".format(epoch, train_acc, test_acc))
        ema.restore(model.parameters())
        model.train()
    torch.save(model.state_dict(), os.path.join(fp, "trained_cnn_weight.pth"))
    return test_acc


def res9_classify(train_set, test_set, fp, num_classes=10, logger=None, reverse_label=False, lr=0.1):
    batch_size = 512
    max_epoch = 50
    criterion = nn.CrossEntropyLoss()
    model = ResNet9(num_classes, 3)
    #model = CNN(img_dim=(3,32,32), num_classes=num_classes).cuda()
    ema = ExponentialMovingAverage(model.parameters(), 0.9999)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, drop_last=False)

    model.train()
    for epoch in range(max_epoch):
        total = 0
        correct = 0
        for _, (inputs, targets) in enumerate(train_loader):
            if len(targets.shape) == 2:
                inputs = (inputs.cuda().to(torch.float32) / 127.5 - 1.)
                targets = torch.argmax(targets, dim=1)
            if reverse_label:
                targets = 1 - targets
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            ema.update(model.parameters())
        train_acc = correct / total * 100
        scheduler.step()
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        total = 0
        correct = 0
        y_list = []
        pred_list = []
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(test_loader):
                if len(targets.shape) == 2:
                    inputs = (inputs.cuda().to(torch.float32) / 127.5 - 1.)
                    targets = torch.argmax(targets, dim=1)
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                y_list.append(targets.detach().cpu())
                pred_list.append(predicted.detach().cpu())
        
        y_list = torch.cat(y_list).numpy()
        pred_list = torch.cat(pred_list).numpy()
        report = classification_report(y_list, pred_list)
        logger.info(report)
        
        test_acc = correct / total * 100

        logger.info("Epoch: {} Train acc: {} Test acc: {}".format(epoch, train_acc, test_acc))
        ema.restore(model.parameters())
        model.train()
    torch.save(model.state_dict(), os.path.join(fp, "trained_cnn_weight.pth"))
    return test_acc

def mlp_classify(train_set, test_set, reverse_label=False):
    model = neural_network.MLPClassifier(max_iter=1000)
    
    train_num = len(train_set)
    test_num = len(test_set)
    train_loader = DataLoader(train_set, batch_size=train_num)
    test_loader = DataLoader(test_set, batch_size=test_num)
    for x_train, y_train in train_loader:
        if len(y_train.shape) == 2:
            x_train = (x_train.cuda().to(torch.float32) / 127.5 - 1.)
            y_train = torch.argmax(y_train, dim=1)
        x_train, y_train = x_train.numpy(), y_train.numpy()
        if reverse_label:
            y_train = 1 - y_train
    for x_test, y_test in test_loader:
        if len(y_test.shape) == 2:
            x_test = (x_test.to(torch.float32) / 127.5 - 1.)
            y_test = torch.argmax(y_test, dim=1)
        x_test, y_test = x_test.numpy(), y_test.numpy()
    x_train = x_train.reshape(train_num, -1)
    x_test = x_test.reshape(test_num, -1)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    return acc


def logreg_classify(train_set, test_set, reverse_label=False):
    model = linear_model.LogisticRegression(solver='lbfgs', max_iter=50000, multi_class='auto')
    
    train_num = len(train_set)
    test_num = len(test_set)
    train_loader = DataLoader(train_set, batch_size=train_num)
    test_loader = DataLoader(test_set, batch_size=test_num)
    for x_train, y_train in train_loader:
        if len(y_train.shape) == 2:
            x_train = (x_train.cuda().to(torch.float32) / 127.5 - 1.)
            y_train = torch.argmax(y_train, dim=1)
        x_train, y_train = x_train.numpy(), y_train.numpy()
        if reverse_label:
            y_train = 1 - y_train
    for x_test, y_test in test_loader:
        if len(y_test.shape) == 2:
            x_test = (x_test.to(torch.float32) / 127.5 - 1.)
            y_test = torch.argmax(y_test, dim=1)
        x_test, y_test = x_test.numpy(), y_test.numpy()
    x_train = x_train.reshape(train_num, -1)
    x_test = x_test.reshape(test_num, -1)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    return acc

def main(args):

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("{}/evaluation_downstream_acc_log.txt".format(args.out_dir), mode='a')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(args)

    if args.c == 1:
        transform_train = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if args.dataset == "mnist":
        test_set = MNIST(root=args.train_dir, train=False, transform=transform_test)
        num_classes = 10
        reverse_label = False
        lr = 3e-4
    elif args.dataset == "fmnist":
        test_set = FashionMNIST(root=args.train_dir, train=False, transform=transform_test)
        num_classes = 10
        reverse_label = False
        lr = 3e-4
    elif args.dataset == "celeba":
        test_set = ImageFolderDataset(args.train_dir, 32, attr='Male', split='test', use_labels=True)
        num_classes = 2
        reverse_label = False
        lr = 3e-4
    elif args.dataset == "camelyon":
        test_set = ImageFolder(root=args.train_dir, transform=transform_test)
        num_classes = 2
        reverse_label = True
        lr = 1e-4
    else:
        raise NotImplementedError

    if 'npz' in args.train_dir:
        data = np.load(args.train_dir)
        x = data['samples']
        y = []
        for i in range(num_classes):
            y += [i] * (len(x) // num_classes)
        y = np.array(y)
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        x = x / 255. * 2. - 1.
        x = x.permute(0, 3, 1, 2)
        if args.c == 1:
            x = transforms.Grayscale(num_output_channels=1)(x)
        x = torch.nn.functional.interpolate(x, size=[args.image_size, args.image_size])
        train_set = TensorDataset(x, y)
    else:
        train_set = ImageFolder(root=args.train_dir, transform=transform_train)
    print(len(train_set))

    if args.model == "cnn":
        acc = cnn_classify(train_set, test_set, args.out_dir, num_classes, logger, reverse_label=reverse_label, lr=lr)
    elif args.model == "res9":
        acc = res9_classify(train_set, test_set, args.out_dir, num_classes, logger, reverse_label=reverse_label)
    elif args.model == "logreg":
        acc = logreg_classify(train_set, test_set, reverse_label=reverse_label)
    elif args.model == "mlp":
        acc = mlp_classify(train_set, test_set, reverse_label=reverse_label)
    else:
        raise NotImplementedError

    logger.info("Final {} acc: {}".format(args.model, acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "mlp", "cnn", "res9"])

    args = parser.parse_args()
    model_list = ["cnn", "logreg", "mlp"]
    for model_ in model_list:
        args.model = model_
        main(args)