class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # defining layers
        self.conv1 = nn.Conv1d(3, 10, 5)
        self.conv2 = nn.Conv1d(10, 15, 5)
        self.conv3 = nn.Conv1d(15, 20, 5)
#         self.conv3 = nn.Conv1d(15, 20, 3)
        self.fc1 = nn.Linear(38 * 20, 12)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 4)
        
        nn.init.xavier_uniform_(self.conv1.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv3.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc1.weight, gain = nn.init.calculate_gain('sigmoid'))
        
    def forward(self, signal):
        signal = torch.transpose(signal, 1, 2)
        out = F.relu(self.conv1(signal))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = torch.transpose(out, 1, 2)
        out = out.reshape(-1, 38 * 20)
        out = F.log_softmax(self.fc1(out), dim = 1)
        return out