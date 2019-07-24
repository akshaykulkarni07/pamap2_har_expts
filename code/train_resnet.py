import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
print(torch.__version__)


# #### Dataset Class (loads data from csv)

reqd_len = 50
channels = 3
classes = 12
class IMUDataset(Dataset):
    def __init__(self, mode = 'test', transform = None):
        if mode == 'train' :
            self.df = pd.read_csv('data/train.csv', header = None)
        elif mode == 'test' :
            self.df = pd.read_csv('data/test.csv', header = None)
        elif mode == 'val' :
            self.df = pd.read_csv('data/val.csv', header = None)
        self.transform = transform
        print(self.df.shape)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        y = self.df.iloc[idx : idx + reqd_len, 3 : ].values
        ind = np.argmax(np.sum(y, axis = 0))
        label = np.zeros_like(self.df.iloc[0, 3 : ].values)
        label = label.astype('float')
        label[ind] = 1
        x = self.df.iloc[idx : idx + reqd_len, : channels].values
        x = x.astype('float')
        x = x.reshape(reqd_len, channels)
        assert(x.shape == (reqd_len, channels))
        assert(label.shape == (classes, ))
        return x, label
        
trainset = IMUDataset(mode = 'train')
valset = IMUDataset(mode = 'val')
testset = IMUDataset(mode = 'test')


# #### DataLoader definitions (provides data in iterable form)

train_batch_size = 64
batch_size = 64
train_indices = [(i * reqd_len) for i in range(len(trainset) // reqd_len)]
val_indices = [(i * reqd_len) for i in range(len(valset) // reqd_len)]
test_indices = [(i * reqd_len) for i in range(len(testset) // reqd_len)]

trainloader = DataLoader(trainset, batch_size = train_batch_size, sampler = SubsetRandomSampler(train_indices), drop_last = True, num_workers = 32)
valloader = DataLoader(valset, batch_size = batch_size, sampler = SubsetRandomSampler(val_indices), drop_last = True, num_workers = 32)
testloader = DataLoader(testset, batch_size = batch_size, sampler = SubsetRandomSampler(test_indices), drop_last = True, num_workers = 32)


# In[4]:

def output_size(n, f, p = 0, s = 1):
    ''' Returns output size for given input size (n), filter size (f), padding (p) and stride (s)
    for a convolutional layer
    '''
    return (((n + 2 * p - f) / s) + 1)

class block(nn.Module):
    def __init__(self, filters, subsample=False):
        super().__init__()
        """
        A 2-layer residual learning building block as illustrated by Fig.2
        in "Deep Residual Learning for Image Recognition"
        
        Parameters:
        
        - filters:   int
                     the number of filters for all layers in this block
                   
        - subsample: boolean
                     whether to subsample the input feature maps with stride 2
                     and doubling in number of filters
                     
        Attributes:
        
        - shortcuts: boolean
                     When false the residual shortcut is removed
                     resulting in a 'plain' convolutional block.
        """
        # Determine subsampling
        s = 0.5 if subsample else 1.0
        
        # Setup layers
        self.conv1 = nn.Conv1d(int(filters*s), filters, kernel_size=3, 
                               stride=int(1/s), padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(filters, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(filters, track_running_stats=True)
        self.relu2 = nn.ReLU()

        # Shortcut downsampling
        self.downsample = nn.AvgPool1d(kernel_size=1, stride=2)

        # Initialise weights according to the method described in 
        # “Delving deep into rectifiers: Surpassing human-level performance on ImageNet 
        # classification” - He, K. et al. (2015)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)   
        
    def shortcut(self, z, x):
        """ 
        Implements parameter free shortcut connection by identity mapping.
        If dimensions of input x are greater than activations then this
        is rectified by downsampling and then zero padding dimension 1
        as described by option A in paper.
        
        Parameters:
        - x: tensor
             the input to the block
        - z: tensor
             activations of block prior to final non-linearity
        """
        if x.shape != z.shape:
            d = self.downsample(x)
            p = torch.mul(d, 0)
            return z + torch.cat((d, p), dim=1)
        else:
            return z + x
    
    def forward(self, x, shortcuts=False):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)
        
        z = self.conv2(z)
        z = self.bn2(z)
        
        # Shortcut connection
        # This if statement is the only difference between
        # a convolutional net and a resnet!
        if shortcuts:
            z = self.shortcut(z, x)

        z = self.relu2(z)
        
        return z
    


class ResNet(nn.Module):
    def __init__(self, n, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts
        
        # Input
        self.convIn = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnIn   = nn.BatchNorm1d(16, track_running_stats=True)
        self.relu   = nn.ReLU()
        
        # Stack1
        self.stack1 = nn.ModuleList([block(16, subsample=False) for _ in range(n)])

        # Stack2
#         self.stack2a = block(32, subsample=True)
#         self.stack2b = nn.ModuleList([block(32, subsample=False) for _ in range(n-1)])

#         # Stack3
#         self.stack3a = block(64, subsample=True)
#         self.stack3b = nn.ModuleList([block(64, subsample=False) for _ in range(n-1)])
        
        # Output
        # The parameters of this average pool are not specified in paper.
        # Initially I tried kernel_size=2 stride=2 resulting in 
        # 64*4*4= 1024 inputs to the fully connected layer. More aggresive
        # pooling as implemented below results in better results and also
        # better matches the total model parameter count cited by authors.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcOut = nn.Linear(1, 12, bias = True)
        self.fcOut2 = nn.Linear(1, 4, bias = True)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # Initilise weights in fully connected layer 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()      
        
    # set flag = True for fine-tuning
    def forward(self, x, flag = False):
        x = torch.transpose(x, 1, 2)
        z = self.convIn(x)
        z = self.bnIn(z)
        z = self.relu(z)
        
        for l in self.stack1: z = l(z, shortcuts=self.shortcuts)
        
#         z = self.stack2a(z, shortcuts=self.shortcuts)
#         for l in self.stack2b: 
#             z = l(z, shortcuts=self.shortcuts)
        
#         z = self.stack3a(z, shortcuts=self.shortcuts)
#         for l in self.stack3b: 
#             z = l(z, shortcuts=self.shortcuts)

        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        if flag : 
            z = self.fcOut2(z)
        else : 
            z = self.fcOut(z)
        return self.softmax(z)

Net = ResNet(n = 3, shortcuts = True)
if torch.cuda.is_available():
    print('Model on GPU')
    Net = Net.cuda()

# Net.load_state_dict(torch.load('saved_models/model3.pt')) 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Net.parameters(), lr = 1e-3)

num_epochs = 200
total_step = len(trainset) // (train_batch_size * reqd_len)
train_loss_list = list()
val_loss_list = list()
min_val = 100
for epoch in range(num_epochs):
    trn = []
    Net.train()
    for i, (images, labels) in enumerate(trainloader) :
        if torch.cuda.is_available():
            images = Variable(images).cuda().float()
            labels = Variable(labels).cuda()
        else : 
            images = Variable(images).float()
            labels = Variable(labels)
        
        _, target = torch.max(labels, 1)

        y_pred = Net(images)
        
        loss = criterion(y_pred, target)
        trn.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0 :
            print('epoch = ', epoch, ' step = ', i, ' of total steps ', total_step, ' loss = ', loss.item())
            
    train_loss = (sum(trn) / len(trn))
    train_loss_list.append(train_loss)
    
    Net.eval()
    val = []
    with torch.no_grad() :
        for i, (images, labels) in enumerate(valloader) :
            if torch.cuda.is_available():
                images = Variable(images).cuda().float()
                labels = Variable(labels).cuda()
            else : 
                images = Variable(images).float()
                labels = Variable(labels)
                
            _, target = torch.max(labels, 1)

            # Forward pass
            outputs = Net(images)
            loss = criterion(outputs, target)
            val.append(loss)

    val_loss = (sum(val) / len(val)).item()
    val_loss_list.append(val_loss)
    print('epoch : ', epoch, ' / ', num_epochs, ' | TL : ', train_loss, ' | VL : ', val_loss)
    
    if val_loss < min_val :
        print('saving model')
        min_val = val_loss
        torch.save(Net.state_dict(), 'saved_models/model6.pt')


# j = np.arange(60)
# plt.plot(j, train_loss_list, 'r', j, val_loss_list, 'g')

def _get_accuracy(dataloader, Net):
    total = 0
    correct = 0
    Net.eval()
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images).float().cuda()
        labels = Variable(labels).float()

        outputs = Net(images)
        
        outputs = outputs.cpu()
    
        _, label_ind = torch.max(labels, 1)
        _, pred_ind = torch.max(outputs, 1)
        
        # converting to numpy arrays
        label_ind = label_ind.data.numpy()
        pred_ind = pred_ind.data.numpy()
        
        # get difference
        diff_ind = label_ind - pred_ind
        # correctly classified will be 1 and will get added
        # incorrectly classified will be 0
        correct += np.count_nonzero(diff_ind == 0)
        total += len(diff_ind)

    accuracy = correct / total
    # print(len(diff_ind))
    return accuracy

Net.cuda()
print(_get_accuracy(trainloader, Net) * 100, '/', _get_accuracy(valloader, Net) * 100, '/', _get_accuracy(testloader, Net) * 100)

testing_Net = ConvNet()
testing_Net.load_state_dict(torch.load('saved_models/model6.pt'))
testing_Net.eval().cuda()
print(_get_accuracy(trainloader, testing_Net) * 100, '/', _get_accuracy(valloader, testing_Net) * 100, '/', _get_accuracy(testloader, testing_Net) * 100)

