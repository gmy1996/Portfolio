import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.l1 = nn.Linear(n_input, n_hidden) 
        self.l2 = nn.Linear(n_hidden, n_hidden) 
        self.l3 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out