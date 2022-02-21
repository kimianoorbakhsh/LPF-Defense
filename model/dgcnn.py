import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.optim as optim


from tqdm.auto import tqdm, trange

import numpy as np

import sklearn.metrics as metrics

import os
import sys
import glob
import h5py



def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = train_data, train_labels
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class DGCNN(nn.Module):
    def __init__(self, k, emb_dims, dropout_p, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.dropout_p = dropout_p
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout_p)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout_p)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

LOG_INTERVAL = 5

def train(model_path = None):
    ##args
    batch_size = 64
    test_batch_size = 16
    num_points = 2048
    k = 20
    emb_dims = 1024
    dropout_p = 0.5
    lr = 0.001
    momentum = 0.9
    use_sgd = True
    start_epoch = 1
    epochs = 200
    ####
    
    train_loader = DataLoader(ModelNet40(partition='train', num_points= num_points),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points= num_points),
                             batch_size=test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running on ", device)


    model = DGCNN(k, emb_dims, dropout_p, output_channels=40).to(device)
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    if use_sgd:
        print("Using SGD")
        opt = optim.SGD(model.parameters(), lr=lr*100, momentum=momentum, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, epochs, eta_min=lr)
    else:
        print("Using Adam")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, epochs, eta_min=lr * 0.01)
        
    criterion = cal_loss
    best_test_acc = 0
    
    
    for epoch in trange(start_epoch, epochs+1, desc='Epochs', leave=True):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        
        for i, data in enumerate(tqdm(train_loader, desc='Batches', leave=False)):
            data, label = data[0].to(device), data[1].to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
            scheduler.step()
            
            
            if i % LOG_INTERVAL == LOG_INTERVAL - 1:
                 print('Train [%d/%d]\t | \tLoss: %.5f' % (i * batch_size, len(train_loader.dataset), loss.item() * batch_size))
        train_loss /= i
        print('==> epoch : %d Train | Average loss: %.5f' % (epoch, train_loss))
        
        ####################
        # Test
        ####################
        
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        
        for i, data in enumerate(test_loader):
            data, label = data[0].to(device), data[1].to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        print('==> Test | loss: %.5f, test acc: %.5f, test avg acc: %.5f' % ( test_loss*1.0/i,
                                                                              test_acc * 100,
                                                                              avg_per_class_acc * 100))
                                                                               
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, 'log/DGCNN_checkpoint_200_noattack_s20.pt')
            #log/DGCNN_checkpoint_150_noattack_s20.pt

def test(model):
    batch_size = 8
    test_batch_size = 16
    num_points = 2048
    test_loader = DataLoader(ModelNet40(partition='test', num_points=num_points),
                             batch_size=test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("testing on ", device)
    model = model.eval()
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    criterion = cal_loss
    for i, data in enumerate(test_loader):
        data, label = data[0].to(device), data[1].to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        loss = criterion(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    print('==> Test | loss: %.5f, test acc: %.5f, test avg acc: %.5f' % ( test_loss*1.0/i,
                                                                          test_acc * 100,
                                                                          avg_per_class_acc * 100))

if torch.cuda.is_available():
      device = torch.device('cuda:0')
      print('running on GPU')
else:
      device = torch.device('cpu') 
      print('running on CPU')
      
# replace the following paths with your own paths
train_data = np.load('Data/no-attack/sigma=20/train_data.npy').astype('float32')
test_data = np.load('Data/no-attack/sigma=20/test_data.npy').astype('float32')
train_labels = np.load('Data/no-attack/sigma=20/train_labels.npy').astype('int64')
test_labels = np.load('Data/no-attack/sigma=20/test_labels.npy').astype('int64')

train()
# uncomment to test the model
# k = 20
# emb_dims = 1024
# dropout_p = 0.5
# lr = 0.001
# momentum = 0.9
# use_sgd = True
# epochs = 250

# model= DGCNN(k, emb_dims, dropout_p, output_channels=40).to(device)
# checkpoint = torch.load('log/DGCNN_checkpoint_s20.pt')
# # device = torch.device('cuda:1')
# model.load_state_dict(checkpoint['model_state_dict'])
# test(model)
