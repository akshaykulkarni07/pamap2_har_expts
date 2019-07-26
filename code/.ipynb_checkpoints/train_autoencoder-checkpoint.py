import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
print(torch.__version__)

reqd_len = 100
channels = 3
class IMUDataset(Dataset):
    def __init__(self, mode = 'test', transform = None):
        if mode == 'train' : 
            self.df = pd.read_csv('data/train.csv', header = None)
        elif mode == 'test' : 
            self.df = pd.read_csv('data/test.csv', header = None)
        self.transform = transform
        print(self.df.shape)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.df.iloc[idx : idx + reqd_len, : channels].values
        x = x.astype('float')
        assert(x.shape == (reqd_len, channels))
        return x
        
train_dataset = IMUDataset(mode = 'train')
test_dataset = IMUDataset(mode = 'test')

batch_size = 32
train_indices = [(i * reqd_len) for i in range(len(train_dataset) // reqd_len)]
test_indices = [(i * reqd_len) for i in range(len(test_dataset) // reqd_len)]

trainloader = DataLoader(train_dataset, batch_size = batch_size, sampler = SubsetRandomSampler(train_indices), drop_last = True)
trainloader2 = DataLoader(train_dataset, batch_size = 1, sampler = SubsetRandomSampler(train_indices), drop_last = True)
testloader2 = DataLoader(test_dataset, batch_size = 1, sampler = SubsetRandomSampler(test_indices), drop_last = True)

# for xavier initialization of network
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
class AutoEncoder(nn.Module) :
    def __init__(self) : 
        super(AutoEncoder, self).__init__()
        # defining layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels = 2, kernel_size = 5),
            nn.Tanh(),
            nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 5),
            nn.Tanh(),
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5),
            nn.Tanh(),
            nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 5),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels = 1, out_channels = 1, kernel_size = 5),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels = 1, out_channels = 2, kernel_size = 5),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels = 2, out_channels = 3, kernel_size = 5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(84, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x, encode = False, classify = False) :
        x = torch.transpose(x, 1, 2)
        features = self.encoder(x)
        
        if encode and not classify:
            return features
        elif not encode and classify :
            features = features.view(-1, 134)
            return self.classifier(features)
        else : 
            return self.decoder(features)
        
Net = AutoEncoder()
Net.apply(init_weights)
if torch.cuda.is_available() : 
    Net = Net.cuda()
    print('Model on GPU')
    
import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(Net.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

num_epochs = 100
total_step = len(train_dataset) // (batch_size * 150)
train_loss_list = list()
min_loss = 100
for epoch in range(num_epochs):
    trn = []
    Net.train()
    for i, signals in enumerate(trainloader) :
        if torch.cuda.is_available():
            signals = Variable(signals).cuda().float()
        else : 
            signals = Variable(signals).float()
        
        reconstr = Net.forward(signals)
        signal_ = torch.transpose(signals, 1, 2).float()
        loss = criterion(reconstr, signal_)
        trn.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
#         torch.nn.utils.clip_grad_value_(Net.parameters(), 10)
        optimizer.step()

        if i % 200 == 0 :
            print('epoch = ', epoch, ' step = ', i, ' of total steps ', total_step, ' loss = ', loss.item())
            
    train_loss = (sum(trn) / len(trn))
    train_loss_list.append(train_loss)
    
    if train_loss < min_loss : 
        min_loss = train_loss
        torch.save(Net.state_dict() , 'saved_models/autoencoder2.pt')
        print('Saving model', min_loss)
    
    scheduler.step()