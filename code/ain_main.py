import numpy as np
import os
from hw2 import preprocessing as P
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F

class Dataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.from_numpy(x).float()
        y = self.labels[index] if self.labels is not None else -1
        return x, y

    def __len__(self):
        return len(self.data)
    
# data loaders
def get_data_loaders(batch_size, train_data, train_labels, dev_data, dev_labels, test_data, test_labels):
    params1 = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2}
    params2 = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2}
    training_set = Dataset(train_data, train_labels)
    train_generator = data.DataLoader(training_set, **params1) 

    dev_set = Dataset(dev_data, dev_labels)
    dev_generator = data.DataLoader(dev_set, **params2) 
    
    testing_set = Dataset(test_data, test_labels)
    test_generator = data.DataLoader(testing_set, **params2) 
    
    print('get_data_loaders success!')
    return (train_generator, dev_generator, test_generator)

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
        self.channel1 = nn.Conv2d(nin, nout, kernel_size = 1, bias = False)
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
        X_out = Xs/Wa
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
            AIL(1024,512, kernel_size = 1, stride = 1, padding = 0),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x




def training_routine(net, n_iters, gpu, train_generator, dev_generator, test_generator):
    if gpu:
        net = net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    best_acc = 0
    
    for i in range(n_iters):
        # Training
        net.train()
        loss_sum = 0
        data_num = 0
        for local_batch, local_labels in train_generator:
            if gpu:
                local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
            train_output = net(local_batch)
            train_prediction = train_output.cpu().detach().argmax(dim=1).numpy()
            train_loss = criterion(train_output, local_labels.long())
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += train_loss.data
            data_num += len(local_batch)
            print(loss_sum/data_num, end = '\r')
        print('[TRAIN]  Epoch [%d]   Loss: %.4f'
                    % (i, loss_sum))    

        # Validate
        with torch.no_grad():
            net.eval()
            acc = 0
            loss_sum = 0
            data_num = 0
            for local_batch, local_labels in dev_generator:
                if gpu:
                    local_batch, local_labels = local_batch.cuda(),local_labels.cuda()
                test_output = net(local_batch)
                test_loss = criterion(test_output, local_labels.long())
                test_prediction = test_output.cpu().detach().argmax(dim=1).numpy()
                acc = acc + (test_prediction == local_labels.cpu().numpy()).sum()
                loss_sum += train_loss.data
                data_num += len(local_batch)
                print(loss_sum/data_num, ' ', acc/data_num,end = '\r')
            print('[Val]  Epoch [%d]   Loss: %.4f  Acc: %.4f'  
                        % (i, loss_sum, acc/data_num)) 
            if acc/data_num > best_acc:
                best_acc = acc/data_num
                print("Saving model, predictions and generated output for epoch " + str(i)+" with acc: " + str(best_acc))
                torch.save(net.state_dict(), 'myModel')  
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
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        
def main():
    train_feats1 = np.load('./train_data.npy')
    train_labels1 = np.load('./train_labels.npy')
    test_feats1 = np.load('./test_data.npy')
    test_labels = np.load('./test_labels.npy')
    train_data1, test_data = P.cifar_10_preprocess(train_feats1, test_feats1)
    
    train_data = train_data1[0:40000]
    train_labels = train_labels1[0:40000]
    dev_data = train_data1[40000:50000]
    dev_labels = train_labels1[40000:50000]

    gpu = torch.cuda.is_available()
    model = AINDense()
    model.apply(weights_init)
    #model.load_state_dict(torch.load('myModel'))
      
    n_iters = 10
    batch_size = 64
    train_generator, dev_generator, test_generator = get_data_loaders(batch_size, train_data, train_labels, dev_data, dev_labels, test_data, test_labels)
    training_routine(model, n_iters, gpu, train_generator, dev_generator, test_generator)
    
if __name__ == '__main__':
    print('main')
    main()


