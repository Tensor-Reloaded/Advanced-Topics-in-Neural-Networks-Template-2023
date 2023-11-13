from multiprocessing import freeze_support

import torch
import torchvision.transforms
from sklearn.decomposition import PCA

import wandb
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from Assignment5 import utils, sam
from Assignment5.ZAE import ZAE
from Assignment5.model import Model
from Assignment5.runner import Runner
from Assignment5.sam import SAM
from Assignment5.transforms_var2 import RandomAugmentation, ToGrayscale, Flatten, RandomRotation, DecomposeChannels, \
    Crop, RecomposeChannels, ColorChange, StandardNormalization, FeatureLabelsSplit, StandardNormalization3
from Assignment5.utils import count_correct_predictions


def sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        batch_size = config.batch_size
        optimizer = config.optimizer
        optimizers = []
        closure = [False]
        if optimizer == 'adam':
            optimizers.append(torch.optim.Adam)
            optimizer_args = [{'lr': config.lr, 'weight_decay': config.weight_decay}]
        elif optimizer == 'sgd':
            optimizers.append(torch.optim.SGD)
            if config.Nesterov:
                momentum = 0.9
            else:
                momentum = config.momentum
            optimizer_args = [{'lr': config.lr, 'weight_decay': config.weight_decay,
                               'nesterov': config.Nesterov, 'momentum': momentum}]
        elif optimizer == 'rmsprop':
            optimizers.append(torch.optim.RMSprop)
            optimizer_args = [{'lr': config.lr, 'weight_decay': config.weight_decay,
                               'momentum': config.momentum}]
        elif optimizer == 'adagrad':
            optimizers.append(torch.optim.Adagrad)
            optimizer_args = [{'lr': config.lr, 'weight_decay': config.weight_decay}]
        elif optimizer == 'sam_sgd':
            optimizers.append(SAM)
            if config.Nesterov:
                momentum = 0.9
            else:
                momentum = config.momentum
            optimizer_args = [{'base_optimizer': torch.optim.SGD,
                               'weight_decay': config.weight_decay,
                               'lr': config.lr, 'momentum': momentum,
                               'nesterov': config.Nesterov}]
            closure = [True]
        if config.lr_scheduler:
            lr_scheduler = {'lr': config.lr, 'epochs': config.epochs}
        else:
            lr_scheduler = None
        batch_normalization = config.batch_norm
        gradient_clipping = config.gradient_clipping
        hidden_layers = [config.fc_layer_size_1, config.fc_layer_size_2, config.fc_layer_size_3]
        dropouts = [config.dropout_1, config.dropout_2, config.dropout_3, config.dropout_4]
        activations = [-1, -1, -1, torch.nn.Softmax(dim=1)]
        model = Model(784, 10, hidden_layers=hidden_layers,
                      device=runner.device, activations=activations, loss=torch.nn.CrossEntropyLoss(),
                      dropouts=dropouts, weight_initialization=[torch.nn.init.xavier_normal_,
                                                                torch.nn.init.xavier_normal_,
                                                                torch.nn.init.xavier_normal_,
                                                                torch.nn.init.xavier_normal_],
                      batch_normalization=batch_normalization, gradient_clipping=gradient_clipping,
                      lr=config.lr, optimizers=optimizers, optimizer_args=optimizer_args,
                      lr_scheduler=lr_scheduler, closure=closure)
    runner.run_model(model, transforms=[v2.ToImage(),
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Resize((28, 28), antialias=True),
                                        # v2.Grayscale(),
                                        # torch.flatten
                                        ],
                     transforms_test=[v2.ToImage(),
                                      v2.ToDtype(torch.float32, scale=True),
                                      v2.Resize((28, 28), antialias=True),
                                      v2.Grayscale(),
                                      torch.flatten
                                      ],
                     dataset_csv=False, load_from_pytorch=True, pin_memory=True,
                     transforms_not_cached=[
                         # RandomAugmentation(instances=[0], num_ops=2, magnitude=9),
                         ToGrayscale(instances=[0]),
                         Flatten(instances=[0]),
                     ], batch_size=batch_size, config=config)


if __name__ == "__main__":
    freeze_support()
    writer = SummaryWriter()

    runner = Runner(1000, utils.get_default_device(),
                    f'.', None,
                    writer, similarity_func=count_correct_predictions,
                    )
    # sweep_id = wandb.sweep(runner.sweep_config, project="pytorch-sweeps-demo")
    # wandb.agent(sweep_id, sweep, count=100)

    # Tensorboard testing + > 60% accuracy
    model = (Model(784, 10, hidden_layers=[4000, 1000, 784, 4000],
                   activations=[torch.nn.ReLU(), -1, -1, torch.nn.ReLU()],
                   optimizers=[SAM], optimizer_args=[{'base_optimizer': torch.optim.SGD,
                                                      'lr': 0.001, 'weight_decay': 0.001,
                                                      'momentum': 0.9
                                                      },
                                                     ],
                   loss=torch.nn.CrossEntropyLoss(), device=utils.get_default_device(), gradient_clipping=True,
                   weight_initialization=[torch.nn.init.xavier_normal_,
                                          torch.nn.init.xavier_normal_,
                                          torch.nn.init.xavier_normal_,
                                          torch.nn.init.xavier_normal_,
                                          torch.nn.init.xavier_normal_,
                                          torch.nn.init.xavier_normal_,
                                          torch.nn.init.xavier_normal_,
                                          torch.nn.init.xavier_normal_,
                                          ],
                   batch_normalization=True, closure=[True],
                   # lr_scheduler={'gamma': 0.001, 'epochs': 312400},
                   dropouts=[torch.nn.Dropout(0.1).to(utils.get_default_device()),
                             torch.nn.Dropout(0.1).to(utils.get_default_device()),
                             torch.nn.Dropout(0.1).to(utils.get_default_device()),
                             torch.nn.Dropout(0.1).to(utils.get_default_device()),
                             torch.nn.Dropout(0.1).to(utils.get_default_device()),
                             ],
                   batch_norms=[torch.nn.BatchNorm1d(784).to(utils.get_default_device()),
                                torch.nn.BatchNorm1d(4000).to(utils.get_default_device()),
                                torch.nn.BatchNorm1d(1000).to(utils.get_default_device()),
                                torch.nn.BatchNorm1d(784).to(utils.get_default_device()),
                                torch.nn.BatchNorm1d(4000).to(utils.get_default_device()),
                                ]
                   )).to(utils.get_default_device())
    checkpoint = torch.load('checkpoint_5')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.activations = [torch.nn.ReLU(), -1, -1, torch.nn.ReLU()]
    model.loss = torch.nn.CrossEntropyLoss()
    model.lr_scheduler = {'epochs': 78100, 'gamma': 0.5}
    runner.run_model(model, transforms=[v2.ToImage(),
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Resize((28, 28), antialias=True),
                                        # v2.Grayscale(),
                                        # torch.flatten,
                                        ],
                     transforms_test=[v2.ToImage(),
                                      v2.ToDtype(torch.float32, scale=True),
                                      v2.Resize((28, 28), antialias=True),
                                      # v2.Grayscale(),
                                      # torch.flatten,
                                      ],
                     dataset_csv=False, load_from_pytorch=True, pin_memory=True,
                     transforms_not_cached=[
                         #StandardNormalization3(instances=[0]),
                         RandomAugmentation(instances=[0], num_ops=2, magnitude=9),
                         ToGrayscale(instances=[0]),
                         Flatten(instances=[0]),
                     ],
                     transforms_not_cached_test=[
                         #StandardNormalization3(instances=[0]),
                         ToGrayscale(instances=[0]),
                         Flatten(instances=[0]),
                     ],
                     batch_size=8192, resume=True)
