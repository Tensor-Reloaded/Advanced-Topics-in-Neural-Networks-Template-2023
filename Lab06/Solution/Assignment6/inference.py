import copy

import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet50_Weights
from torchvision.transforms import v2

from Assignment6 import utils
from Assignment6.dataset import Dataset
from Assignment6.model import Model
from Assignment6.plotter import MetricsMemory
from Assignment6.pyramid_net import PyramidNet
from Assignment6.resnet import ResNet
from Assignment6.resnet_custom import ResnetCustom


def validate(code, device=utils.get_default_device()):
    # load model and its meta-info depending on the code
    if code not in ['10', '10_p', '100', '100_p', 'own']:
        raise ValueError('Code not available')
    if code == '10':
        no_class = 10
        transforms_test = [torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])]
        model = PyramidNet(optimizers=[torch.optim.SGD],
                           optimizer_args=[{'lr': 0.25, 'weight_decay': 1e-4,
                                            'momentum': 0.9,
                                            'nesterov': True
                                            },
                                           ],
                           closure=[False],
                           loss=torch.nn.CrossEntropyLoss(),
                           device=utils.get_default_device(),
                           lr_scheduler=MultiStepLR,
                           lr_scheduler_args={'milestones': [150, 225],
                                              'gamma': 0.1},
                           alpha=1.0,
                           dataset='cifar10',
                           depth=240,
                           num_classes=10
                           )
        model.load_state_dict(torch.load('checkpoint_10_5')['model_state_dict'])
        val_dataset = CIFAR10(root='../data', train=False,
                              transform=v2.Compose(transforms_test), download=True)
        val_dataset = Dataset('../data',
                              lambda path: (tuple([x for x in val_dataset]), len(val_dataset)),
                              transformations=[], transformations_test=[], training=False)
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0,
                                batch_size=128, drop_last=False)
    elif code == '10_p':
        no_class = 10
        transforms_test = [torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010)),
                           ]
        model_res = copy.deepcopy(torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2))
        model = ResnetCustom(model_res,
                             optimizers=[torch.optim.SGD],
                             optimizer_args=[{'lr': 0.01, 'weight_decay': 1e-4,
                                              'momentum': 0.9,
                                              'nesterov': True
                                              },
                                             ],
                             closure=[False],
                             loss=torch.nn.CrossEntropyLoss(),
                             device=utils.get_default_device(),
                             lr_scheduler=MultiStepLR,
                             lr_scheduler_args={'milestones': [150 * 781, 225 * 781],
                                                'gamma': 0.1}
                             )
        model.load_state_dict(torch.load('checkpoint_10_2_p')['model_state_dict'],
                              strict=False)
        val_dataset = CIFAR10(root='../data', train=False,
                              transform=v2.Compose(transforms_test), download=True)
        val_dataset = Dataset('../data',
                              lambda path: (tuple([x for x in val_dataset]), len(val_dataset)),
                              transformations=[], transformations_test=[], training=False)
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0,
                                batch_size=128, drop_last=False)
    elif code == 'own':
        no_class = 10
        transforms_test = [torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])]
        model = Model(optimizers=[torch.optim.SGD],
                      optimizer_args=[{'lr': 0.01, 'weight_decay': 1e-4,
                                       'momentum': 0.9,
                                       'nesterov': True
                                       },
                                      ],
                      closure=[False],
                      loss=torch.nn.CrossEntropyLoss(),
                      device=utils.get_default_device(),
                      lr_scheduler=MultiStepLR,
                      lr_scheduler_args={'milestones': [150 * 781, 225 * 781],
                                         'gamma': 0.1},
                      layers=[
                          torch.nn.Conv2d(in_channels=3, out_channels=50, kernel_size=(3, 3), stride=1, padding='same'),
                          torch.nn.ReLU(),
                          torch.nn.Conv2d(in_channels=50, out_channels=64, kernel_size=(3, 3), stride=1,
                                          padding='same'),
                          torch.nn.ReLU(),
                          torch.nn.MaxPool2d(kernel_size=(2, 2)),
                          torch.nn.Dropout(0.2),
                          torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1,
                                          padding='same'),
                          torch.nn.ReLU(),
                          torch.nn.MaxPool2d(kernel_size=(2, 2)),
                          torch.nn.Dropout(0.2),
                          torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1,
                                          padding='same'),
                          torch.nn.ReLU(),
                          torch.nn.MaxPool2d(kernel_size=(2, 2)),
                          torch.nn.Dropout(0.2),
                          torch.nn.Flatten(),
                          torch.nn.Linear(4096, 500),
                          torch.nn.ReLU(),
                          torch.nn.Dropout(0.2),
                          torch.nn.Linear(500, 250),
                          torch.nn.ReLU(),
                          torch.nn.Dropout(0.2),
                          torch.nn.Linear(250, 10),
                          torch.nn.Softmax(dim=1)
                      ],
                      to_initialize=[0, 2, 6, 10, 15, 18, 21],
                      weight_initialization=[torch.nn.init.xavier_normal_,
                                             torch.nn.init.xavier_normal_,
                                             torch.nn.init.xavier_normal_,
                                             torch.nn.init.xavier_normal_,
                                             torch.nn.init.xavier_normal_,
                                             torch.nn.init.xavier_normal_,
                                             torch.nn.init.xavier_normal_,
                                             ]
                      )
        model.load_state_dict(torch.load('checkpoint_simpler')['model_state_dict'])

        val_dataset = CIFAR10(root='../data', train=False,
                              transform=v2.Compose(transforms_test), download=True)
        val_dataset = Dataset('../data',
                              lambda path: (tuple([x for x in val_dataset]), len(val_dataset)),
                              transformations=[], transformations_test=[], training=False)
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0,
                                batch_size=128, drop_last=False)
    elif code == '100':
        no_class = 100
        transforms_test = [torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])]
        model = PyramidNet(optimizers=[torch.optim.SGD],
                           optimizer_args=[{'lr': 0.1, 'weight_decay': 1e-4,
                                            'momentum': 0.9,
                                            'nesterov': True
                                            },
                                           ],
                           closure=[False],
                           loss=torch.nn.CrossEntropyLoss(),
                           device=utils.get_default_device(),
                           lr_scheduler=MultiStepLR,
                           lr_scheduler_args={'milestones': [150, 225],
                                              'gamma': 0.1},
                           alpha=1.0,
                           dataset='cifar100',
                           depth=240,
                           num_classes=100
                           )
        model.load_state_dict(torch.load('checkpoint_100_78')['model_state_dict'])

        val_dataset = CIFAR100(root='../data', train=False,
                               transform=v2.Compose(transforms_test), download=True)
        val_dataset = Dataset('../data',
                              lambda path: (tuple([x for x in val_dataset]), len(val_dataset)),
                              transformations=[], transformations_test=[], training=False)
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0,
                                batch_size=128, drop_last=False)

    similarity = utils.count_correct_predictions
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            correct += (similarity(outputs, torch.nn.functional.one_hot(
                labels, num_classes=no_class).float()))
            total_loss += model.loss(outputs, torch.nn.functional.one_hot(
                labels, num_classes=no_class).float()).item()
    print(f'Loss: {total_loss / (len(val_loader) * val_loader.batch_size)}, '
          f'Accuracy: {correct / (len(val_loader) * val_loader.batch_size)}\n')


if __name__ == '__main__':
    validate('10_p')

# {10, 10_p, 100, 100_p, own} reprezinta coduri pentru checkpoint-urile CIFAR10 PyramidNet, CIFAR10 preantrenat,
# CIFAR100 PyramidNet, CIFAR100 preantrenat, own_cnn
