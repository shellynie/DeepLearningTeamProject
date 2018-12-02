import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader
import time

import numpy as np 
import os
from tensorboard_logging import *

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALIDATION_SPLIT = .2
RANDOM_SEED = 42
SHUFFLE_DATASET = True
NUM_WORKERS = 0
np.random.seed(42)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
cifar_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

print(len(trainset))

class cifar_mask_dataset(Dataset):
    def __init__(self, indices):
        self.data = [trainset[i] for i in indices]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

dataset_size = len(trainset)
indices = list(range(dataset_size))
split = int(np.floor(VALIDATION_SPLIT * dataset_size))
if SHUFFLE_DATASET :
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

cifar_train_dataset = cifar_mask_dataset(train_indices)
cifar_dev_dataset = cifar_mask_dataset(val_indices)


cifar_train_loader = torch.utils.data.DataLoader(cifar_train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=NUM_WORKERS)
cifar_dev_loader = torch.utils.data.DataLoader(cifar_dev_dataset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=NUM_WORKERS)

cifar_test_loader = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



print(len(cifar_train_loader), len(cifar_train_dataset))
print(len(cifar_dev_loader), len(cifar_dev_dataset))
print(len(cifar_test_loader), len(cifar_test_loader))

class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove the last layer
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(2048, 10)

    def forward(self, x):
        self.resnet.eval()
        out = self.resnet(x)
        return self.classifier(out.view(out.size(0), -1))


class Densenet121(nn.Module):
    def __init__(self):
        super(Densenet121, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.embedding = densenet.features
        for p in self.embedding.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x):
        self.embedding.eval()
        features = self.embedding(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        return self.classifier(out.view(out.size(0), -1))



class Trainer():
    def __init__(self, model, optimizer, criterion, name, 
        trainLoader, devLoader, load_path=None, scheduler=None):
        self.model = model
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
        self.optimizer = optimizer
        self.criterion = criterion
        self.name = name
        self.trainLoader = trainLoader
        self.devLoader = devLoader

    def save_model(self):
        torch.save(self.model.state_dict(), get_model_path(self.name))

    def run(self, n_epochs):
        print("Start Training...")
        tLog = Logger("./logs/train_{0}".format(self.name))

        for i in range(n_epochs):
            start_time = time.time()
            epoch_loss = 0
            correct = 0

            for i_batch, (batch_data, batch_label) in enumerate(self.trainLoader):
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()

                self.optimizer.zero_grad()
                X = Variable(batch_data)
                Y = Variable(batch_label)

                self.model.train()
                trainOutput = self.model(X)
                pred = trainOutput.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                correct += predicted.sum()

                trainLoss = self.criterion(trainOutput, Y)
                trainLoss.backward()
                self.optimizer.step()
                epoch_loss += trainLoss.data.item()

            train_size = len(cifar_train_dataset)
            train_loss = epoch_loss / train_size
            train_acc = correct.data.item() / train_size
            val_loss, val_acc = inference(self.model, self.criterion, self.devLoader)
            print("epoch: {0}, train loss: {1:.8f}, train acc: {2:.8f}, val loss: {3:.8f}, val acc: {4:.8f}".format(i+1, train_loss, train_acc, val_loss, val_acc))
            tLog.log_scalar("train_loss", train_loss, i+1)
            tLog.log_scalar("train_acc", train_acc, i+1)
            tLog.log_scalar("val_loss", val_loss, i+1)
            tLog.log_scalar("val_acc", val_acc, i+1)
            for tag, value in self.model.named_parameters():
                tag.replace('.', '/')
                tLog.log_histogram(tag, value.data.cpu().numpy(), i+1)
                # tLog.log_histogram(tag+'/grad', value.grad.data.cpu().numpy(), i+1)
            if (i+1) % 5 == 0:
                torch.save(self.model.state_dict(), "./{0}_model_epoch{1}.pt".format(self.name, i+1))
            end_time = time.time()
            print("time used: {0}s".format(end_time - start_time))

gpu = True and torch.cuda.is_available()
if gpu:
    print("Using GPU")
else:
    print("Using CPU")


def get_model_path(name):
    return "./{0}_model.pt".format(name)


def get_prediction_path(name):
    return "./{0}_pred.npy".format(name)


def inference(model, criterion, devLoader):
    correct = 0
    loss = 0
    dev_size = len(cifar_dev_dataset)
    for i_batch, (batch_data, batch_label) in enumerate(devLoader):
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()

        X = Variable(batch_data)
        Y = Variable(batch_label)
        model.eval()
        out = model(X)
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
        loss += criterion(out, Y).data.item()
    return (loss / dev_size, correct.data.item() / dev_size)


def predict(mlp, name, loader):
    load_path = get_model_path(name)
    save_path = get_prediction_path(name)
    trainer = Trainer(mlp, None, None, name, None, None, load_path)
    with open(save_path, "w") as f:
        f.write("id,label\n")
        counter = 0
        for i_batch, (batch_data, batch_label) in enumerate(loader):
            print(i_batch)
            batch_data = batch_data.cuda()
            X = Variable(batch_data)
            trainer.model.eval()
            out = trainer.model(X)
            pred = out.data.max(1, keepdim=True)[1]
            data = pred.data.cpu().numpy()
            for i in range(data.shape[0]):
                f.write("{0},{1}\n".format(counter, data[i][0]))
                counter += 1


def train_model1():
    name = "baseline_resnet101_1"
    # load_path = "baseline11_model_epoch17_err0.04889741.pt"
    # trainer = Trainer(resnet_3.cuda(), None, None, name, None, load_path)
    # model = trainer.model.cuda()
    model = Resnet101().cuda()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, nesterov=True, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, name, cifar_train_loader, cifar_dev_loader, scheduler=scheduler)
    trainer.run(20)
    trainer.save_model()

def train_model2():
    name = "baseline_densenet121_1"
    model = Densenet121().cuda()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, nesterov=True, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, name, cifar_train_loader, cifar_dev_loader, scheduler=scheduler)
    trainer.run(20)
    trainer.save_model()

if __name__ == "__main__":
    train_model2()
    # predict(Resnet101().cuda(), "baseline_resnet101_1", cifar_test_loader)
    print("check")
