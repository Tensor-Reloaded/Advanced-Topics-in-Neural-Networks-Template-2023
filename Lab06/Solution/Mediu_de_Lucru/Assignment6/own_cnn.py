from multiprocessing import freeze_support

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Assignment6 import utils
from Assignment6.model import Model
from Assignment6.runner import Runner
from Assignment6.utils import count_correct_predictions

if __name__ == "__main__":
    freeze_support()
    writer = SummaryWriter()

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
                      torch.nn.Conv2d(in_channels=50, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
                      torch.nn.ReLU(),
                      torch.nn.MaxPool2d(kernel_size=(2, 2)),
                      torch.nn.Dropout(0.2),
                      torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
                      torch.nn.ReLU(),
                      torch.nn.MaxPool2d(kernel_size=(2, 2)),
                      torch.nn.Dropout(0.2),
                      torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding='same'),
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

    model_forward = torch.jit.trace(model, torch.randn(64, 3, 32, 32).to(utils.get_default_device()))
    # model_forward = torch.jit.script(model, torch.randn(64, 3, 32, 32).to(utils.get_default_device()))

    runner = Runner(300, utils.get_default_device(),
                    f'.', None,
                    writer, similarity_func=count_correct_predictions,
                    )

    runner.run_model(model, model_forward=model, transforms=[
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])],
                     transforms_test=[transforms.ToTensor(),
                                      transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                           std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                                      ],
                     dataset_csv=False,
                     load_from_pytorch=True, pin_memory=True,
                     transforms_not_cached=[
                     ], batch_size=64, num_workers=4, num_classes=10)
