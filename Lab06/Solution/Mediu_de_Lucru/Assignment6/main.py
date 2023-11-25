import copy
from multiprocessing import freeze_support

import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet50_Weights
from torchvision.transforms import transforms, v2
from torchvision.transforms.v2 import ColorJitter

from Assignment6 import utils
from Assignment6.Handmade_Conv2d import Handmade_conv2d_implementation
from Assignment6.pyramid_net import PyramidNet
from Assignment6.resnet import ResNet
from Assignment6.resnet_custom import ResnetCustom
from Assignment6.runner import Runner
from Assignment6.transforms_var2 import Lighting, MinMaxNormalization, RandomAugmentation
from Assignment6.utils import count_correct_predictions

if __name__ == "__main__":
    inp = torch.randn(1, 3, 10, 12)  # Input image
    # kernel of size 4x5, with 3 input channels and 2 output channels
    w = torch.randn(2, 3, 4, 5)  # Conv weights
    # My implementation
    custom_conv_2d_layer = Handmade_conv2d_implementation(weights=w)
    out = custom_conv_2d_layer(inp)
    print((torch.nn.functional.conv2d(inp, w) - out).abs().max())

    freeze_support()
    writer = SummaryWriter()

    # model = PyramidNet(optimizers=[torch.optim.SGD],
    #                    optimizer_args=[{'lr': 0.25, 'weight_decay': 1e-4,
    #                                     'momentum': 0.9,
    #                                     'nesterov': True
    #                                     },
    #                                    ],
    #                    closure=[False],
    #                    loss=torch.nn.CrossEntropyLoss(),
    #                    device=utils.get_default_device(),
    #                    lr_scheduler=MultiStepLR,
    #                    lr_scheduler_args={'milestones': [150, 225],
    #                                       'gamma': 0.1},
    #                    alpha=1.0,
    #                    dataset='cifar100',
    #                    depth=240,
    #                    num_classes=100
    #                    )

    model_res = copy.deepcopy(torchvision.models.resnet50(pretrained=True))
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
                                            'gamma': 0.1},
                         )

    runner = Runner(300, utils.get_default_device(),
                    f'.', None,
                    writer, similarity_func=count_correct_predictions,
                    )

    # runner.run_model(model, transforms=[transforms.RandomCrop(32, padding=4),
    #                                     transforms.RandomHorizontalFlip(),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])],
    #                  transforms_test=[transforms.ToTensor(),
    #                                   transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    #                                   ],
    #                  dataset_csv=False, load_from_pytorch=True, pin_memory=True,
    #                  transforms_not_cached=[
    #                  ], batch_size=64, num_workers=4, num_classes=10)

    runner.run_model(model, transforms=[
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ColorJitter(brightness=0.4, contrast=0.4,
        #             saturation=0.4),
        # Lighting(alphastd=0.1,
        #          eigval=[0.2175, 0.0188, 0.0045],
        #          eigvec=[[-0.5675, 0.7192, 0.4009],
        #                  [-0.5808, -0.0045, -0.8140],
        #                  [-0.5836, -0.6948, 0.4203]]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        # transforms.Resize((224, 224), antialias=True)
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ],
                     transforms_test=[
                         # transforms.Resize(256),
                         # transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         #                      std=[0.229, 0.224, 0.225]),
                         # transforms.Resize((224, 224), antialias=True)
                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010)),
                     ],
                     dataset_csv=False, load_from_pytorch=True, pin_memory=True,
                     transforms_not_cached=[
                     ], batch_size=64, num_workers=4, num_classes=100)
