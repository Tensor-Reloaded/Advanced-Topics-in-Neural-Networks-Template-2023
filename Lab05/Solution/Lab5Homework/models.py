import torch
import torch.nn as nn

__all__ = ['CifraMLP']


class CifraMLP(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim):
        super(CifraMLP, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(input_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, output_dim)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.nn.functional.elu(self.fc1(x), alpha=1)
        x = self.dropout1(x)
        x = torch.nn.functional.elu(self.fc2(x), alpha=1)
        x = self.dropout2(x)
        x = torch.nn.functional.elu(self.fc3(x), alpha=1)
        x = self.fc4(x)
        return x
