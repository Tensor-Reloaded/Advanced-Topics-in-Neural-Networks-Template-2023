# Testing module
import torch

from Assignment5 import utils
from Assignment5.runner import Runner
from Assignment5.transforms import FeatureLabelsSplit, MinMaxNormalization, NumberToTensor, ImageToTensor, ChangeType, \
    ReshapeTensors
from Assignment5.utils import build_dataset_initial

runner = Runner(100, utils.get_default_device(),
                f'.\\Homework_Dataset', build_dataset_initial)

runner.build_model(transforms=[NumberToTensor(instances=[2]),
                               ImageToTensor(instances=[0, 1]),
                               ChangeType(dtype=torch.float32),
                               MinMaxNormalization(instances=[0, 1], minim=0, maxim=255),
                               MinMaxNormalization(instances=[2], minim=0, maxim=12),
                               ReshapeTensors(128 * 128 * 3, instances=[0, 1]),
                               FeatureLabelsSplit(features=[0, 2], labels=[1])],
                   split_path=f'.\\HD_Split', dataset_csv=True)
