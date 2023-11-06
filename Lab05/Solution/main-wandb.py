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
hidden_layers = [512, 256, 128]
output_size = 10 
epochs = 10
nesterov = True
momentum = 0.9

activation_fns = [torch.nn.ReLU() for _ in hidden_layers]
    
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

wandb_project_name = "RN.LAB5"
wandb_sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'validation_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-5,
            'max': 1e-2
        },
        'batch_size': {
            'values': [64, 128, 256, 512]
        },
        'optimizer':{
            'values': ['adam', 'sgd', 'rmsprop', 'adagrad', 'samsgd']
        },
    }
}

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


cutmix = v2.CutMix(num_classes=output_size)
mixup = v2.MixUp(num_classes=output_size)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

def wandb_run():   
    with wandb.init():

        #swept hyperparams
        learning_rate = wandb.config.learning_rate
        training_batch_size = wandb.config.batch_size
        optimizer_name = wandb.config.optimizer

        model = Model(input_size, hidden_layers, output_size, activation_fns)
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(optimizer_name, model=model, learning_rate=learning_rate, momentum=momentum)

        trainloader = DataLoader(trainset, batch_size=training_batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=512, shuffle=False)


        logger_name = f".XEntropy+Adam.lr={learning_rate}.batch_size={training_batch_size}"
        logger = Logger(tensorboard_file_suffix=logger_name)
        trainer = Trainer(model, criterion, optimizer, logger=logger)
        trainer.run(trainloader, testloader, epochs)

def main():
    wandb.login()
    sweep_id = wandb.sweep(wandb_sweep_config, project=wandb_project_name)
    wandb.agent(sweep_id, function=wandb_run)  

if(__name__ == "__main__"):
    main()