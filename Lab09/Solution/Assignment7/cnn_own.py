from multiprocessing import freeze_support

import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import VGG16_Weights
from torchvision.transforms import v2

from Assignment7 import utils
from Assignment7.model import Model
from Assignment7.runner import Runner
from Assignment7.transforms_var2 import RandomAugmentation
from Assignment7.utils import apply_cosine_similarity

if __name__ == "__main__":
    freeze_support()
    writer = SummaryWriter()
    model = Model(optimizers=[torch.optim.Adam],
                  optimizer_args=[{'lr': 0.0002
                                   }],
                  closure=[False],
                  loss=torch.nn.MSELoss(),
                  device=utils.get_default_device(),
                  lr_scheduler=MultiStepLR,
                  lr_scheduler_args={'milestones': [150 * 781, 225 * 781],
                                     'gamma': 0.1},
                  layers=[
                      torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=1, padding=(1, 1)),
                      torch.nn.ReLU(),
                      torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=1, padding=(1, 1)),
                      torch.nn.ReLU(),
                      torch.nn.Dropout(0.2),
                      torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=1, padding=(1, 1)),
                      torch.nn.ReLU(),
                      torch.nn.Dropout(0.2),
                      torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(4, 4), stride=1, padding=(1, 1)),
                  ],
                  to_initialize=[0, 2, 5, 8],
                  weight_initialization=[torch.nn.init.xavier_normal_,
                                         torch.nn.init.xavier_normal_,
                                         torch.nn.init.xavier_normal_,
                                         torch.nn.init.xavier_normal_,
                                         ]
                  )

    runner = Runner(300, utils.get_default_device(),
                    f'.', None,
                    writer, similarity_func=apply_cosine_similarity,
                    )

    runner.run_model(model, transforms=[
                                        v2.ToImage(),
                                        v2.ToDtype(torch.float32, scale=True)
                                       ],
                     transforms_test=[
                                        v2.ToImage(),
                                        v2.ToDtype(torch.float32, scale=True)
                                      ],
                     dataset_csv=False, load_from_pytorch=True, pin_memory=True,
                     transforms_not_cached=[
                         # RandomAugmentation(instances=[0], num_ops=2, magnitude=11)
                     ], batch_size=64, num_workers=4)
