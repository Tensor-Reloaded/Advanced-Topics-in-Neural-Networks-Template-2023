import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from Assignment7 import utils
from Assignment7.dataset import Dataset
from Assignment7.model import Model
from Assignment7.test_inference_time import test_inference_time
from Assignment7.utils import build_transformed_dataset


def validate(device=utils.get_default_device()):
    # load model and its meta-info depending on the code
    # model = Model(optimizers=[torch.optim.Adam],
    #               optimizer_args=[{'lr': 0.0002
    #                                }],
    #               closure=[False],
    #               loss=torch.nn.MSELoss(),
    #               device=utils.get_default_device(),
    #               lr_scheduler=MultiStepLR,
    #               lr_scheduler_args={'milestones': [150 * 781, 225 * 781],
    #                                  'gamma': 0.1},
    #               layers=[
    #                   torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=1, padding=(1, 1)),
    #                   torch.nn.ReLU(),
    #                   torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=1, padding=(1, 1)),
    #                   torch.nn.ReLU(),
    #                   torch.nn.Dropout(0.2),
    #                   torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=1, padding=(1, 1)),
    #                   torch.nn.ReLU(),
    #                   torch.nn.Dropout(0.2),
    #                   torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(4, 4), stride=1, padding=(1, 1)),
    #               ],
    #               to_initialize=[0, 2, 5, 8],
    #               weight_initialization=[torch.nn.init.xavier_normal_,
    #                                      torch.nn.init.xavier_normal_,
    #                                      torch.nn.init.xavier_normal_,
    #                                      torch.nn.init.xavier_normal_,
    #                                      ]
    #               )
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
                      torch.nn.Flatten(),
                      torch.nn.Linear(3 * 1024, 128),
                      torch.nn.Linear(128, 784),
                      torch.nn.Sigmoid()
                  ],
                  to_initialize=[1, 2],
                  weight_initialization=[torch.nn.init.xavier_normal_,
                                         torch.nn.init.xavier_normal_,
                                         ]
                  )
    model.load_state_dict(torch.load('checkpoint_1st')['model_state_dict'])
    for m in model.modules():
        m = m.to(torch.device('cpu'))
    t1_total = 0
    t2_total = 0
    for trial in range(30):
        t1, t2 = test_inference_time(model)
        t1_total += t1
        t2_total += t2
    print(t1_total / 30)
    print(t2_total / 30)

    # this is for output images saving
    # transforms_test = [v2.ToImage(),
    #                    v2.ToDtype(torch.float32, scale=True)]
    # val_dataset = CIFAR10(root='../data', train=False,
    #                       transform=v2.Compose(transforms_test), download=True)
    # val_dataset = Dataset(val_dataset,
    #                       build_transformed_dataset,
    #                       transformations=[], transformations_test=[], training=False,
    #                       save=False)
    # val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0,
    #                         batch_size=1, drop_last=False)
    # model.eval()
    # index = 0
    # with torch.no_grad():
    #     for features, labels in val_loader:
    #         features = features.to(device)
    #         outputs = model(features)
    #         # save outputs
    #         torchvision.utils.save_image(outputs.reshape(28, 28), f'Outputs\\output_{index}.png')
    #         index += 1


if __name__ == '__main__':
    validate()

# {10, 10_p, 100, 100_p, own} reprezinta coduri pentru checkpoint-urile CIFAR10 PyramidNet, CIFAR10 preantrenat,
# CIFAR100 PyramidNet, CIFAR100 preantrenat, own_cnn
