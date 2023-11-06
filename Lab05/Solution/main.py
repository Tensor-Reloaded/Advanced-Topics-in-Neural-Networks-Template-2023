import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from model import Model, Trainer
from datasets import CachedDataset

input_size = 28 * 28
hidden_layers = [512, 256, 128]
output_size = 10 
activation_fns = [torch.nn.ReLU() for _ in hidden_layers]
learning_rate = 0.001
epochs = 10

model = Model(input_size, hidden_layers, output_size, activation_fns)
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

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

trainer = Trainer(model, criterion, optimizer)
trainer.run(trainloader, testloader, epochs)
