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
from torch.autograd import Variable

NB_CLASSES = 128
IMAGE_SIZE = 224


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
                    self.idx2id.append(idx)  
                    self.labels[idx] = id_label

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx = self.idx2id[idx]
        p = f'./data/{self.preffix}/{idx}.jpg'
        try:
            img = Image.open(p)
            w, h= img.size
            max_len = max(w, h)
            valid_w = int(w * 1.0 / max_len * 224)
            valid_h = int(h * 1.0 / max_len * 224)
            img = transforms.Resize((valid_h, valid_w))(img)
            img = transforms.Pad(padding = (0,0, 224-valid_w, 224-valid_h))(img)
            img = self.transform(img)
        except:
            return torch.zeros((3, 224, 224)), 200, (0,0)

        c, w, h = img.shape
        if c != 3 or w != 224 or h != 224:
            return torch.zeros((3, 224, 224)), 200, (0,0)

        target = self.labels[idx] if len(self.labels) > 0 else -1
        
        return img, target, (valid_w, valid_h)
    
def collate(batch):
    new_img = []
    new_target = []
    new_size = []
    new_h = []
    for img, target, valid_size in batch:
        if target == 200:
            continue
        c, w, h = img.shape
        new_img.append(img.view(1, c, w, h))
        new_target.append(target)
        new_size.append(valid_size)
    return torch.cat(new_img), torch.from_numpy(np.array(new_target)), new_size

def preprocess_crop():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class AIL_MASK(nn.Module):
    def __init__(self, nin, nout, kernel_size = (3,3), stride = 2, padding = 1):
        super(AIL_MASK, self).__init__()
        self.channel1 = nn.Conv2d(nin, nout, kernel_size = 1)
        self.relu = nn.ReLU(True)
        self.channel2 = depthwise_separable_conv(nin, nout)
        self.sigmoid = nn.Sigmoid()
        self.kernel_size = kernel_size
        self.window = nn.AvgPool2d(kernel_size = kernel_size, stride = stride, padding = padding)

    def forward(self, x_in, size_list): #(c,M,N)
        X = self.channel1(x_in)
        X = self.relu(X)
        W = self.channel2(x_in)
        W = self.sigmoid(W) # b,c,M,N
        # mask
        mask = Variable(W.data.new(W.size(0), W.size(1),W.size(2), W.size(3)).zero_(), requires_grad=False)
        for i, size in enumerate(size_list):
            w, h = size
            mask[i, :, :h, :w] = 1
        W = W * mask  

        Xc = W * X
        # shifting window
        Xs = self.window(Xc) 
        Wa = self.window(W)
        X_out = Xs/(Wa + 1e-16)
        return X_out

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
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), stride = 2, padding = 3, bias=False),   
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.ain1 = AIL_MASK(64,64)
        self.conv2 = nn.Sequential(
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
    def forward(self, x, size_list=None):
        x = self.conv1(x)
        x = self.ain1(x, size_list)
        x = self.conv2(x)
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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-6)
    best_acc = 0
    
    for i in range(n_iters):
        # Training
        net.train()
        loss_sum = 0
        data_num = 0
        for local_batch, local_labels, local_size in train_generator:
            if gpu:
                local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
            train_output = net(local_batch, local_size)
            #train_prediction = train_output.cpu().detach().argmax(dim=1).numpy()
            train_loss = criterion(train_output, local_labels.long())
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += train_loss.data
            data_num += len(local_batch)
            print(loss_sum/data_num, end = '\r')
        print('[TRAIN]  Epoch [%d]   Loss: %.4f    Data_num: %d'
                    % (i, loss_sum, data_num)) 
        torch.save(net.state_dict(), 'myModel3') 

        # Validate
        with torch.no_grad():
            net.eval()
            acc = 0
            loss_sum = 0
            data_num = 0
            for local_batch, local_labels, local_size in dev_generator:
                if gpu:
                    local_batch, local_labels = local_batch.cuda(),local_labels.cuda()
                test_output = net(local_batch, local_size)
                test_loss = criterion(test_output, local_labels.long())
               
                test_prediction = test_output.cpu().detach().argmax(dim=1).numpy()
                acc = acc + (test_prediction == local_labels.cpu().numpy()).sum()
                
                loss_sum += test_loss.data
                data_num += len(local_batch)
                print(loss_sum/data_num, ' ', acc/data_num,end = '\r')
            print('[Val]  Epoch [%d]   Loss: %.4f  Acc: %.4f'  
                        % (i, loss_sum, acc/data_num)) 
            if acc/data_num > best_acc:
                best_acc = acc/data_num
                print("Saving model, predictions and generated output for epoch " + str(i)+" with acc: " + str(best_acc))
                torch.save(net.state_dict(), 'myModel3best')   

    net = net.cpu() 
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    



def main():
    gpu = torch.cuda.is_available()
    model = AINDense()
    model.apply(weights_init)
    #model.load_state_dict(torch.load('myModel3'))
      
    n_iters = 1
    batch_size = 128

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
    

    training_routine(model, n_iters, gpu, train_generator, dev_generator)
    
if __name__ == '__main__':
    print('main')
    main()

