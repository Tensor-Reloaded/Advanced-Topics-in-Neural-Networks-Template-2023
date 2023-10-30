import torch
import torch.nn as nn


__all__ = ['SimpleModel']

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        
        
        self.fc1 = nn.Linear(input_dim, input_dim//16)
        self.fc2 = nn.Linear(input_dim//16, input_dim//32)
        self.fc3 = nn.Linear(input_dim//32, output_dim//64)
        self.fc4 = nn.Linear(output_dim//64,output_dim//32)
        self.fc5 = nn.Linear(output_dim//32,output_dim//32)
        self.fc6 = nn.Linear(output_dim//32,output_dim//16)
        self.fc7 = nn.Linear(output_dim//16+1,output_dim)

    def forward(self, x, time):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        #print(x.shape)
        time=time.reshape(x.shape[0],1)
        x = torch.concatenate((x, time), axis=-1)
        #print(x.shape)
        x = self.fc7(x)
        #x= 255 * (x - torch.min(x, dim=1, keepdim=True)[0]) / (torch.max(x, dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0])
        return x