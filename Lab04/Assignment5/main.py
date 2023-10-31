# Testing module
import cv2
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2

from Assignment5 import utils
from Assignment5.model import Model
from Assignment5.runner import Runner
from Assignment5.transforms import FeatureLabelsSplit, MinMaxNormalization, NumberToTensor, ImageToTensor, ChangeType, \
    ReshapeTensors, RandomRotation, GroupTensors, ColorChange, UngroupTensors, Crop, DecomposeChannels, \
    RecomposeChannels
from Assignment5.utils import build_dataset_initial

runner = Runner(100, utils.get_default_device(),
                f'.\\Homework_Dataset', build_dataset_initial)

model = (Model(128 * 128 * 3 + 1, 128 * 128 * 3, hidden_layers=[128, 128],
               activations=[-1, -1, torch.sigmoid],
               optimizers=[torch.optim.Adam], optimizer_args=[{'lr': 0.001}],
               loss=torch.nn.MSELoss(), device=utils.get_default_device(),
               dropouts=[('a', 0.5)], gradient_clipping=True,
               weight_initialization=[torch.nn.init.xavier_normal,
                                      torch.nn.init.xavier_normal,
                                      torch.nn.init.xavier_normal],
               batch_normalization=True)).to(utils.get_default_device())
runner.run_model(model, transforms=[NumberToTensor(instances=[2]),
                                    ImageToTensor(instances=[0, 1]),
                                    ChangeType(dtype=torch.float32),
                                    # DecomposeChannels(instances=[0, 1]),
                                    # Crop(instances=[0, 1],
                                    #      shape=torch.Size([128, 128])),  # random transformation, correct
                                    # RecomposeChannels(instances=[0, 1]),
                                    GroupTensors(instances=[0, 1]),
                                    DecomposeChannels(instances=[0]),
                                    # ColorChange(instances=[0]),  # random transformation, correct
                                    RandomRotation(instances=[0]),  # random transformation, correct
                                    RecomposeChannels(instances=[0]),
                                    UngroupTensors(instance=0, dim=128),
                                    MinMaxNormalization(instances=[2], minim=0, maxim=12),
                                    ReshapeTensors(128 * 128 * 3, instances=[0, 1]),
                                    FeatureLabelsSplit(features=[0, 2], labels=[1])],
                 split_path=f'.\\HD_Split', dataset_csv=True,
                 transforms_test=[NumberToTensor(instances=[2]),
                                  ImageToTensor(instances=[0, 1]),
                                  ChangeType(dtype=torch.float32),
                                  MinMaxNormalization(instances=[2], minim=0, maxim=12),
                                  ReshapeTensors(128 * 128 * 3, instances=[0, 1]),
                                  FeatureLabelsSplit(features=[0, 2], labels=[1])])
