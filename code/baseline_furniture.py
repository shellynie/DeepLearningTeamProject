import warnings
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader
import time
from tensorboardX import SummaryWriter

import numpy as np 
import os

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from Dataset import Furniture128, collate_fn

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

ROOT_DIR = '/home/xiejie/project'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CKPT_DIR = os.path.join(ROOT_DIR, 'ckpts')
NUM_EPOCHS = 80
BATCH_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
RANDOM_SEED = 42
SHUFFLE_DATASET = True
NUM_WORKERS = 0
LR = 0.005
IMAGE_SIZE = 224
LOG_FREQ = 5
CKPT_FREQ = 4

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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

furniture_train_dataset = Furniture128(DATA_DIR, 'train', transform)
furniture_dev_dataset = Furniture128(DATA_DIR, 'valid', transform)

furniture_train_loader = DataLoader(furniture_train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)
furniture_dev_loader = DataLoader(furniture_dev_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_fn)

print('train size: {}'.format(len(furniture_train_dataset)))
print('dev size: {}'.format(len(furniture_dev_dataset)))

class Resnet101(nn.Module):
    def __init__(self, init=False):
        super(Resnet101, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove the last layer
        for p in self.resnet.parameters():
            if not init:
                p.requires_grad = False
            else:
                p.requires_grad = True
        self.classifier = nn.Linear(2048, 128)

    def forward(self, x):
        out = self.resnet(x)
        return self.classifier(out.view(out.size(0), -1))


class Densenet121(nn.Module):
    def __init__(self, init=False):
        super(Densenet121, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.embedding = densenet.features
        for p in self.embedding.parameters():
            if not init:
                p.requires_grad = False
            else:
                p.requires_grad = True
        self.classifier = nn.Linear(1024, 128)

    def forward(self, x):
        features = self.embedding(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        return self.classifier(out.view(out.size(0), -1))


class Trainer():
    def __init__(self, model, optimizer, criterion, name, 
        trainLoader, devLoader, scheduler, load_path=None):
        self.model = model
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
        self.optimizer = optimizer
        self.criterion = criterion
        self.name = name
        self.trainLoader = trainLoader
        self.devLoader = devLoader
        self.scheduler = scheduler
        self.logger = SummaryWriter()
        self.best_model_path = None

    def run(self, n_epochs):
        best_acc = 0
        best_epoch = 0
        for i in range(n_epochs):
            torch.cuda.empty_cache()
            start_time = time.time()
            epoch_loss = 0
            # correct = 0
            num_nonempty_batches = 0
            num_samples = 0
            for i_batch, (batch_data, batch_label) in enumerate(self.trainLoader):
                # the batch is empty after ignoring invalid images
                if batch_data is None:
                    continue
                num_nonempty_batches += 1
                batch_size = batch_data.size(0)
                num_samples += batch_size
                self.optimizer.zero_grad()
                self.model.train()
                trainOutput = self.model(batch_data)
                # pred = trainOutput.data.max(1, keepdim=True)[1]
                # predicted = pred.eq(batch_label.view_as(pred))
                # batch_correct = predicted.sum().item()
                # correct += batch_correct
                trainLoss = self.criterion(trainOutput, batch_label)
                trainLossVal = trainLoss.data.item()
                trainLoss.backward()
                self.optimizer.step()
                epoch_loss += trainLossVal
                if i_batch % LOG_FREQ == 0:
                    print('Batch ({0}): {1}, Train Loss: {2:.4f}'.format(batch_size, i_batch, epoch_loss / num_nonempty_batches))
            train_loss = epoch_loss / num_nonempty_batches
            # train_acc = correct / num_samples
            val_loss, val_acc, val_num_samples = inference(self.model, self.criterion, self.devLoader)
            print("Epoch: {0}, Train Loss ({1}): {2:.8f}, Val Loss ({3}): {4:.8f}, Val Acc: {5:.8f}".format(i+1, num_samples, train_loss, val_num_samples, val_loss, val_acc))
            hours, rem = divmod(time.time() - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print('epoch train time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
            self.logger.add_scalar("train/loss", train_loss, i+1)
            # self.logger.add_scalar("train/acc", train_acc, i+1)
            self.logger.add_scalar("val/loss", val_loss, i+1)
            self.logger.add_scalar("val/acc", val_acc, i+1)
            self.scheduler.step(val_acc)
            for tag, value in self.model.named_parameters():
                tag.replace('.', '/')
                self.logger.add_histogram(tag, value.data.cpu().numpy(), i+1)
                # self.logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), i+1)
            if val_acc > best_acc and i % CKPT_FREQ == 0:
                best_acc = val_acc
                best_epoch = i + 1
                self.best_model_path = os.path.join(CKPT_DIR, "{0}_model_epoch{1}_valacc{2:.4f}.pt".format(self.name, best_epoch, best_acc))
                torch.save(self.model.state_dict(), self.best_model_path)
                print("Saved ckpts to {}".format(self.best_model_path))
            end_time = time.time()
            print("time used: {0}s".format(end_time - start_time))

def inference(model, criterion, devLoader):
    correct = 0
    loss = 0
    dev_size = 0
    num_nonempty_batches = 0
    num_samples = 0
    with torch.no_grad():
        model.eval()
        for i_batch, (batch_data, batch_label) in enumerate(devLoader):
            if batch_data is None:
                continue
            num_nonempty_batches += 1
            num_samples += batch_data.size(0)
            out = model(batch_data)
            pred = out.data.max(1, keepdim=True)[1]
            predicted = pred.eq(batch_label.view_as(pred))
            correct += predicted.sum().item()
            loss += criterion(out, batch_label).data.item()
        return (loss / num_nonempty_batches, correct / num_samples, num_samples)


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
    name = "baseline_resnet101"
    model = Resnet101(init=True).to(DEVICE)
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs.'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)
    # scheduler = MultiStepLR(optimizer, milestones=[38, 53], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, name, train_loader, dev_loader, scheduler=scheduler)
    trainer.run(NUM_EPOCHS)

def train_model2(train_loader, dev_loader):
    name = "baseline_densenet121"
    model = Densenet121(init=True).to(DEVICE)
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs.'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1)
    # scheduler = MultiStepLR(optimizer, milestones=[38, 53], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, name, train_loader, dev_loader, scheduler=scheduler)
    trainer.run(NUM_EPOCHS)

if __name__ == "__main__":
    train_model2(furniture_train_loader, furniture_dev_loader)
    print('All finished!')
