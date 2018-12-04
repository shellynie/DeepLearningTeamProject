import warnings
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader
import time
from tensorboardX import SummaryWriter

import numpy as np 
import os
# from tensorboard_logging import *

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from Dataset import Furniture128, collate_fn

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

ROOT_DIR = '/home/xiejie/project'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CKPT_DIR = os.path.join(ROOT_DIR, 'ckpts')
NUM_EPOCHS = 80
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = 'cpu'
VALIDATION_SPLIT = .2
RANDOM_SEED = 42
SHUFFLE_DATASET = True
NUM_WORKERS = 0
LR = 0.005
IMAGE_SIZE = 224
LOG_FREQ = 5
np.random.seed(42)

gpu = DEVICE == 'cuda'
if gpu:
    print("Using GPU")
else:
    print("Using CPU")

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize,
# ])

def preprocess_crop():
    # return transforms.Compose([
    #     transforms.RandomCrop(IMAGE_SIZE, padding=IMAGE_SIZE // 8),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize(IMAGE_SIZE),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


transform = preprocess_crop()

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# cifar_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)

# print(len(trainset))

# class cifar_mask_dataset(Dataset):
#     def __init__(self, indices):
#         self.data = [trainset[i] for i in indices]

#     def __getitem__(self, i):
#         return self.data[i]

#     def __len__(self):
#         return len(self.data)

# dataset_size = len(trainset)
# indices = list(range(dataset_size))
# split = int(np.floor(VALIDATION_SPLIT * dataset_size))
# if SHUFFLE_DATASET :
#     np.random.seed(RANDOM_SEED)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# cifar_train_dataset = cifar_mask_dataset(train_indices)
# cifar_dev_dataset = cifar_mask_dataset(val_indices)
furniture_train_dataset = Furniture128(DATA_DIR, 'train', transform)
furniture_dev_dataset = Furniture128(DATA_DIR, 'valid', transform)


# cifar_train_loader = torch.utils.data.DataLoader(cifar_train_dataset, batch_size=BATCH_SIZE,
#                                           shuffle=True, num_workers=NUM_WORKERS)
# cifar_dev_loader = torch.utils.data.DataLoader(cifar_dev_dataset, batch_size=BATCH_SIZE,
#                                         shuffle=False, num_workers=NUM_WORKERS)

# cifar_test_loader = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=BATCH_SIZE,
#                                          shuffle=False, num_workers=NUM_WORKERS)

furniture_train_loader = DataLoader(furniture_train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)
furniture_dev_loader = DataLoader(furniture_dev_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# print(len(cifar_train_loader), len(cifar_train_dataset))
# print(len(cifar_dev_loader), len(cifar_dev_dataset))
# print(len(cifar_test_loader), len(cifar_test_loader))
print('train size: {}'.format(len(furniture_train_dataset)))
print('dev size: {}'.format(len(furniture_dev_dataset)))

class Resnet101(nn.Module):
    def __init__(self, init=False):
        super(Resnet101, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove the last layer
        # if not for initilization, freeze all layers except for the last linear layer
        if not init:
            for p in self.resnet.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(2048, 128)

    def forward(self, x):
        self.resnet.eval()
        out = self.resnet(x)
        return self.classifier(out.view(out.size(0), -1))


class Densenet121(nn.Module):
    def __init__(self, init=False):
        super(Densenet121, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.embedding = densenet.features
        # if not for initilization, freeze all layers except for the last linear layer
        if not init:
            for p in self.embedding.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(1024, 128)

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
        self.logger = SummaryWriter()
        self.best_model_path = None

    # def save_model(self):
    #     torch.save(self.model.state_dict(), get_model_path(self.name))

    def run(self, n_epochs):
        print("Start Training...")
        # tLog = Logger("./logs/train_{0}".format(self.name))
        best_acc = 0
        best_epoch = 0
        for i in range(n_epochs):
            torch.cuda.empty_cache()
            start_time = time.time()
            epoch_loss = 0
            correct = 0
            num_nonempty_batches = 0
            num_samples = 0
            for i_batch, (batch_data, batch_label) in enumerate(self.trainLoader):
                # the batch is empty after ignoring invalid images
                if batch_data is None:
                    continue
                num_nonempty_batches += 1
                self.optimizer.zero_grad()
                X = Variable(batch_data)
                Y = Variable(batch_label)
                num_samples += X.size(0)

                self.model.train()
                trainOutput = self.model(X)
                pred = trainOutput.data.max(1, keepdim=True)[1]
                predicted = pred.eq(Y.data.view_as(pred))
                batch_correct = predicted.sum().item()
                batch_size = predicted.size(0)
                correct += batch_correct

                trainLoss = self.criterion(trainOutput, Y)
                trainLossVal = trainLoss.data.item()
                trainLoss.backward()
                self.optimizer.step()
                epoch_loss += trainLossVal
                if i_batch % LOG_FREQ == 0:
                    print('batch: {0}, train loss: {1:.4f}, train acc: {2:.4f}'.format(i_batch, trainLossVal, batch_correct / batch_size))
            train_loss = epoch_loss / num_nonempty_batches
            train_acc = correct.data.item() / num_samples
            val_loss, val_acc = inference(self.model, self.criterion, self.devLoader)
            print("epoch: {0}, train loss: {1:.8f}, train acc: {2:.8f}, val loss: {3:.8f}, val acc: {4:.8f}".format(i+1, train_loss, train_acc, val_loss, val_acc))
            hours, rem = divmod(time.time() - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print('epoch train time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
            self.logger.add_scalar("train_loss", train_loss, i+1)
            self.logger.add_scalar("train_acc", train_acc, i+1)
            self.logger.add_scalar("val_loss", val_loss, i+1)
            self.logger.add_scalar("val_acc", val_acc, i+1)
            for tag, value in self.model.named_parameters():
                tag.replace('.', '/')
                self.logger.add_histogram(tag, value.data.cpu().numpy(), i+1)
                # self.logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), i+1)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = i + 1
                self.best_model_path = os.path.join(CKPT_DIR, "{0}_model_epoch{1}_valacc{2}.pt".format(self.name, best_epoch, best_acc))
                torch.save(self.model.state_dict(), self.best_model_path)
                print("Saved ckpts to {}".format(self.best_model_path))
            end_time = time.time()
            print("time used: {0}s".format(end_time - start_time))


# def get_model_path(name):
#     return os.path.join(CKPT_DIR, "{0}_model_epoch.pt".format(name))


# def get_prediction_path(name):
#     return "./{0}_pred.npy".format(name)


def inference(model, criterion, devLoader):
    correct = 0
    loss = 0
    dev_size = 0
    num_nonempty_batches = 0
    num_samples = 0
    for i_batch, (batch_data, batch_label) in enumerate(devLoader):
        if batch_data is None:
            continue
        num_nonempty_batches += 1
        X = Variable(batch_data)
        Y = Variable(batch_label)
        num_samples += X.size(0)
        model.eval()
        out = model(X)
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
        loss += criterion(out, Y).data.item()
    return (loss / num_nonempty_batches, correct.data.item() / num_samples)


# def predict(mlp, name, loader, load_path):
#     # load_path = get_model_path(name)
#     save_path = get_prediction_path(name)
#     trainer = Trainer(mlp, None, None, name, None, None, load_path)
#     with open(save_path, "w") as f:
#         f.write("id,label\n")
#         counter = 0
#         for i_batch, (batch_data, batch_label) in enumerate(loader):
#             print(i_batch)
#             batch_data = batch_data.to(DEVICE)
#             X = Variable(batch_data)
#             trainer.model.eval()
#             out = trainer.model(X)
#             pred = out.data.max(1, keepdim=True)[1]
#             data = pred.data.cpu().numpy()
#             for i in range(data.shape[0]):
#                 f.write("{0},{1}\n".format(counter, data[i][0]))
#                 counter += 1


def train_model1(train_loader, dev_loader):
    name = "baseline_resnet101_1"
    # load_path = "baseline11_model_epoch17_err0.04889741.pt"
    # trainer = Trainer(resnet_3.cuda(), None, None, name, None, load_path)
    # model = trainer.model.cuda()
    model = Resnet101(init=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    scheduler = MultiStepLR(optimizer, milestones=[38, 53], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, name, train_loader, dev_loader, scheduler=scheduler)
    trainer.run(NUM_EPOCHS)
    # trainer.save_model()

def train_model2(train_loader, dev_loader):
    name = "baseline_densenet121_1"
    model = Densenet121(init=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, nesterov=True, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    scheduler = MultiStepLR(optimizer, milestones=[38, 53], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, name, train_loader, dev_loader, scheduler=scheduler)
    trainer.run(NUM_EPOCHS)
    # trainer.save_model()

if __name__ == "__main__":
    train_model2(furniture_train_loader, furniture_dev_loader)
    # predict(Resnet101().cuda(), "baseline_resnet101_1", cifar_test_loader)
    print("check")
