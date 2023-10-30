import torch
import torch.nn as nn

class PossiblyWork(nn.Module):
    def __init__(self, channels, output_activation=None):
        super(PossiblyWork, self).__init__()
        self.output_activation = output_activation if output_activation is not None else nn.Identity()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(513, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, channels, kernel_size=1)
        )
        
        self.add_number = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, months):
        x = self.encoder(x)
        
        feature = months.view(months.size(0), months.size(1), 1, 1)
        feature = feature.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat((x, feature), dim=1)
        x = self.decoder(x)
        
        return self.output_activation(x)