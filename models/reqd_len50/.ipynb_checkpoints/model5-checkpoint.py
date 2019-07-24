class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # defining layers
        self.conv1 = nn.Conv1d(3, 5, 3)
        self.conv2 = nn.Conv1d(5, 10, 3)
        self.fc1 = nn.Linear(46 * 10, 256)
        self.fc2 = nn.Linear(256, 64)
        self.pamap = nn.Linear(64, 12)
        self.robogame = nn.Linear(64, 4)
        
        nn.init.xavier_uniform_(self.conv1.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc1.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.pamap.weight, gain = nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.robogame.weight, gain = nn.init.calculate_gain('sigmoid'))
        
    # use flag = True during fine-tuning 
    def forward(self, signal, flag = False):
        signal = torch.transpose(signal, 1, 2)
        out = F.relu(self.conv1(signal))
        out = F.relu(self.conv2(out))
        out = torch.transpose(out, 1, 2)
        out = out.reshape(-1, 46 * 10)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        if flag : 
            out = F.log_softmax(self.robogame(out), dim = 1)
        else :
            out = F.log_softmax(self.pamap(out), dim = 1)
        return out