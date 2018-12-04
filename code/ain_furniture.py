import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import skimage.io
import csv
from PIL import Image
from torchvision import transforms

NB_CLASSES = 128
IMAGE_SIZE = 224
#TRAIN_DEF = '../data/data/train/1.jpg'
#VAL_DEF = './data/valid/1.jpg'
#TEST_DEF = './data/test/1.jpg'
#DEF = {}
#DEF['train'] = TRAIN_DEF
#DEF['valid'] = VAL_DEF
#DEF['test'] = TEST_DEF

class FurnitureDataset(Dataset):
    def __init__(self, preffix: str, transform=None):
        self.preffix = preffix
        self.idx2id = []
        self.labels = {}
        if preffix != 'test':
            with open(f'./data/{preffix}.csv', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    idx = int(row['image_id'])
                    id_label = int(row['label_id']) - 1
                    #if (id_label >= 50):
                    #    continue
                    self.idx2id.append(idx)  
                    self.labels[idx] = id_label

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx = self.idx2id[idx]
        #idx  = idx
        p = f'./data/{self.preffix}/{idx}.jpg'
        try:
            img = Image.open(p)
            img = self.transform(img)
        except:
            return torch.zeros((3, 224, 224)), 200
        '''
            img = self.transform(Image.open(DEF[self.preffix]))
            idx = 1

        c, w, h = img.shape
        if c != 3 or w != 224 or h != 224:
        	img = self.transform(Image.open(DEF[self.preffix]))
        	idx = 1
        '''

        c, w, h = img.shape
        if c != 3 or w != 224 or h != 224:
            return torch.zeros((3, 224, 224)), 200

        target = self.labels[idx] if len(self.labels) > 0 else -1

        return img, target

def collate(batch):
    new_img = []
    new_target = []
    for img, target in batch:
        if target == 200:
            continue
        c, w, h = img.shape
        new_img.append(img.view(1, c, w, h))
        new_target.append(target)
    return torch.cat(new_img), torch.from_numpy(np.array(new_target))

 
def preprocess_crop():
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
'''
def preprocess_crop():
    return transforms.Compose([
        #transforms.Resize(IMAGE_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        #transforms.Normalize(
        #    mean=[0.5, 0.5, 0.5],
        #    std=[0.5, 0.5, 0.5]
        #)
    ])
'''
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class AIL(nn.Module):
    def __init__(self, nin, nout, kernel_size = (3,3), stride = 2, padding = 1):
        super(AIL, self).__init__()
        self.channel1 = nn.Conv2d(nin, nout, kernel_size = 1)
        self.relu = nn.ReLU(True)
        self.channel2 = depthwise_separable_conv(nin, nout)
        self.sigmoid = nn.Sigmoid()
        self.kernel_size = kernel_size
        self.window = nn.AvgPool2d(kernel_size = kernel_size, stride = stride, padding = padding)

    def forward(self, x_in): #(c,M,N)
        X = self.channel1(x_in)
        X = self.relu(X)
        W = self.channel2(x_in)
        W = self.sigmoid(W)
        Xc = W * X
        # shifting window
        Xs = self.window(Xc) 
        Wa = self.window(W)
        X_out = Xs/(Wa + 1e-16)
        return X_out
    
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
            
            
class AINDense(nn.Module):
    def __init__(self):
        super(AINDense, self).__init__()
        # CNN Layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), stride = 2, padding = 3, bias=False),   
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            AIL(64,64),
            _DenseBlock(num_layers = 6, num_input_features = 64, bn_size = 4, growth_rate = 32, drop_rate = 0.2),
            AIL(256,64),
            _DenseBlock(num_layers = 12, num_input_features = 64, bn_size = 4, growth_rate = 16, drop_rate = 0.2),
            AIL(256,128),
            _DenseBlock(num_layers = 24, num_input_features = 128, bn_size = 4, growth_rate = 16, drop_rate = 0.2),
            AIL(512,256),
            _DenseBlock(num_layers = 16, num_input_features = 256, bn_size = 4, growth_rate = 48, drop_rate = 0.2),
            AIL(1024,512, kernel_size = 7, stride = 1, padding = 0),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x

def training_routine(net, n_iters, gpu, train_generator, dev_generator):
    if gpu:
        net = net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002, weight_decay=1e-6)
    best_acc = 0.29

    #net.load_state_dict(torch.load('myModel'))

    for i in range(n_iters):
        
        print('Start training...')
        # Training
        net.train()
        loss_sum = 0
        data_num = 0
        for b, (local_batch, local_labels) in enumerate(train_generator):
            if gpu:
                torch.cuda.empty_cache()
                local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
            train_output = net(local_batch)
            #train_prediction = train_output.cpu().detach().argmax(dim=1).numpy()
            train_loss = criterion(train_output, local_labels.long())
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += train_loss.data
            data_num += len(local_batch)
            print(b)
            print(loss_sum/data_num, end = '\r')
        print('[TRAIN]  Epoch [%d]   Loss: %.4f    Data_num: %d'
                    % (i, loss_sum, data_num))  

        print("Saving model for epoch " + str(i))
        torch.save(net.state_dict(), 'myModel2')  
    

        # Validate
        with torch.no_grad():
            print('Start validating...')
            net.eval()
            acc = 0
            loss_sum = 0
            data_num = 0
            for local_batch, local_labels in dev_generator:
                if gpu:
                    torch.cuda.empty_cache()
                    local_batch, local_labels = local_batch.cuda(),local_labels.cuda()
                test_output = net(local_batch)
                test_loss = criterion(test_output, local_labels.long())
                '''
                valid_pred = torch.argmax(test_output, dim=1)
                predicted = valid_pred.eq(local_labels)
                acc += predicted.sum()
                '''
                test_prediction = test_output.cpu().argmax(dim=1).numpy()
                acc = acc + (test_prediction == local_labels.cpu().numpy()).sum()
                
                loss_sum += test_loss.data
                data_num += len(local_batch)
                print(loss_sum/data_num, ' ', acc/data_num, end = '\r')
            print('[Val]  Epoch [%d]   Loss: %.4f  Acc: %.4f'  
                        % (i, loss_sum, acc/data_num)) 
            if acc/data_num > best_acc:
                best_acc = acc/data_num
                print("Saving model, predictions and generated output for epoch " + str(i)+" with acc: " + str(best_acc))
                torch.save(net.state_dict(), 'myModel_acc')  
    '''
    # test
    with torch.no_grad():
        net.eval()
        acc = 0
        data_num = 0
        for local_batch, local_labels in test_generator:
            if gpu:
                local_batch, local_labels = local_batch.cuda(),local_labels.cuda()
            test_output = net(local_batch)
            test_prediction = test_output.cpu().detach().argmax(dim=1).numpy()
            acc = acc + (test_prediction == local_labels.cpu().numpy()).sum()
            data_num += len(local_batch)
            print(acc/data_num,end = '\r')
        print('[Test]  Acc: %.4f'  
                        % (acc/data_num))  

    net = net.cpu()
    '''
    
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


def main():
    gpu = torch.cuda.is_available()
    model = AINDense()
    #model.apply(weights_init)
    model.load_state_dict(torch.load('myModel4'))
      
    n_iters = 5
    batch_size = 256

    train_generator = DataLoader(
        dataset = FurnitureDataset('train', transform=preprocess_crop()),
        num_workers = 0,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate
    )
    
    dev_generator = DataLoader(
        dataset = FurnitureDataset('valid', transform=preprocess_crop()),
        num_workers = 0,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate
    )
    '''
    test_generator = DataLoader(
        dataset = FurnitureDataset('valid', transform=preprocess_crop()),
        num_workers = 0,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate
    )
    '''

    training_routine(model, n_iters, gpu, train_generator, dev_generator)
    
if __name__ == '__main__':
    print('main')
    main()

