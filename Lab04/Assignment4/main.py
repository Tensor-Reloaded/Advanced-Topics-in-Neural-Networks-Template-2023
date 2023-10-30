# Testing module
import cv2
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2

from Assignment4 import utils
from Assignment4.model import Model
from Assignment4.runner import Runner
from Assignment4.transforms import FeatureLabelsSplit, MinMaxNormalization, NumberToTensor, ImageToTensor, ChangeType, \
    ReshapeTensors, RandomRotation, GroupTensors, ColorChange, UngroupTensors, Crop, DecomposeChannels, \
    RecomposeChannels
from Assignment4.utils import build_dataset_initial

runner = Runner(100, utils.get_default_device(),
                f'.\\Homework_Dataset', build_dataset_initial)

model = (Model(128 * 128 * 3 + 1, 128 * 128 * 3, hidden_layers=[128, 128],
               activations=[-1, -1, torch.sigmoid],
               optimizers=[torch.optim.Adam], optimizer_args=[{'lr': 0.001}],
               loss=torch.nn.MSELoss(), device=utils.get_default_device())
               # dropouts=[('a', 0.2), ('a', 0.2)], weight_initialization=[torch.nn.init.xavier_normal,
               #                                                           torch.nn.init.xavier_normal,
               #                                                           torch.nn.init.xavier_normal],
               # batch_normalization=True)
         .to(utils.get_default_device()))
runner.run_model(model, transforms=[NumberToTensor(instances=[2]),
                                    ImageToTensor(instances=[0, 1]),
                                    ChangeType(dtype=torch.float32),
                                    # DecomposeChannels([0, 1]),
                                    # Crop(instances=list(range(6))),
                                    # ColorChange(instances=list(range(6))),  # random transformation, correct
                                    # RandomRotation(instances=[0, 1]),  # random transformation, correct
                                    # RecomposeChannels(),
                                    # MinMaxNormalization(instances=[0, 1], minim=0, maxim=255),
                                    # MinMaxNormalization(instances=[2], minim=0, maxim=12),
                                    ReshapeTensors(128 * 128 * 3, instances=[0, 1]),
                                    FeatureLabelsSplit(features=[0, 2], labels=[1])],
                 split_path=f'.\\HD_Split', dataset_csv=True,
                 transforms_test=[NumberToTensor(instances=[2]),
                                  ImageToTensor(instances=[0, 1]),
                                  ChangeType(dtype=torch.float32),
                                  MinMaxNormalization(instances=[0, 1], minim=0, maxim=255),
                                  MinMaxNormalization(instances=[2], minim=0, maxim=12),
                                  ReshapeTensors(128 * 128 * 3, instances=[0, 1]),
                                  FeatureLabelsSplit(features=[0, 2], labels=[1])])
