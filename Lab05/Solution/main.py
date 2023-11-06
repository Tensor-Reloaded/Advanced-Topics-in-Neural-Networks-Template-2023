from logger import Logger
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from model import Model, Trainer
from datasets import CachedDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device =  torch.device('mos')
else:
    device = torch.device('cpu')

input_size = 28 * 28
hidden_layers = [512, 256, 128]
output_size = 10 
activation_fns = [torch.nn.ReLU() for _ in hidden_layers]
learning_rate = 0.001
epochs = 100

model = Model(input_size, hidden_layers, output_size, activation_fns)
model.to_device(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

transforms = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((28, 28), interpolation=v2.InterpolationMode.BILINEAR, antialias=False),
    v2.Grayscale(),
    torch.flatten,
])

trainset = CachedDataset(CIFAR10(root='./data', train=True, download=True, transform=transforms))
testset = CachedDataset(CIFAR10(root='./data', train=False, download=True, transform=transforms))

trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
testloader = DataLoader(testset, batch_size=512, shuffle=False)

logger = Logger(tensorboard_file_suffix="XEntropy+Adam")
trainer = Trainer(model, criterion, optimizer, logger=logger)
trainer.run(trainloader, testloader, epochs)
