import wandb
from logger import Logger
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader, default_collate
from sam import SAM

from model import Model, Trainer
from datasets import CachedDataset, WithTransform

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device =  torch.device('mos')
else:
    device = torch.device('cpu')

#constants
input_size = 28 * 28
# hidden_layers = [512, 256, 128]
output_size = 10 
epochs = 120
nesterov = True
momentum = 0.9

cachedTransforms = v2.Compose([
        v2.ToPILImage(),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), interpolation=v2.InterpolationMode.BILINEAR, antialias=False),
        v2.Grayscale(),
    ])

randomTransforms = v2.Compose([
        v2.RandAugment(),
    ])

finalTransforms = v2.Compose([
        torch.flatten,
    ])

trainset = WithTransform(
                CachedDataset(CIFAR10(root='./data', train=True, download=True, transform=cachedTransforms)), 
                transform=v2.Compose([*randomTransforms.transforms, *finalTransforms.transforms])
            )

testset = CachedDataset(CIFAR10(root='./data', train=False, download=True, transform=v2.Compose([*cachedTransforms.transforms, *finalTransforms.transforms])))

def get_optimizer(name:str, model, learning_rate, momentum):
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=nesterov, momentum=momentum)
    if name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    if name == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    if name == 'samsgd':
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=nesterov, momentum=momentum)
        return SAM(model.parameters(), inner_optimizer)
    
    raise ValueError("Unknown optimizer")


def main():
    wandb.login()
    wandb.init()

    learning_rate = 0.0026704041957093883 # params from best wandb run
    training_batch_size = 128
    optimizer_name = 'adam'
    hidden_layers = [512, 256, 128]

    activation_fns = [torch.nn.ReLU() for _ in hidden_layers]
    model = Model(input_size, hidden_layers, output_size, activation_fns)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model=model, learning_rate=learning_rate, momentum=momentum)

    trainloader = DataLoader(trainset, batch_size=training_batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=512, shuffle=False)


    log_file_suffix = f".optimizer={optimizer_name}.lr={learning_rate}.batch_size={training_batch_size}"
    logger = Logger(tensorboard_file_suffix=log_file_suffix, log_to_wandb=True)

    logger.log_text_config('Criterion', "cross entropy")
    logger.log_text_config('Optimizer', optimizer_name)
    logger.log_scalar_config('Training batch size', training_batch_size)
    logger.log_scalar_config('Learning rate', learning_rate)

    trainer = Trainer(model, criterion, optimizer, logger=logger)
    trainer.run(trainloader, testloader, epochs)

if(__name__ == "__main__"):
    main()