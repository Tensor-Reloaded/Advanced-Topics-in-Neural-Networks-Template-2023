# Project Summary - Custom ResNet34 for CIFAR-10 and CIFAR-100

## Overview
This project involved implementing a custom ResNet34 architecture suitable for classifying images from the CIFAR-10 and CIFAR-100 datasets. It includes a detailed examination of the architecture, a focus on transfer learning, and performance optimizations using PyTorch's JIT compilation.

## Implementation Details
- Custom ResNet34 model was developed to closely follow the architecture of torchvision models.
- Transfer learning was employed by initializing the custom model with weights from the pretrained torchvision ResNet34 model.
- The model was adapted for CIFAR datasets by adjusting the final fully connected layer to match the number of target classes.
- Feature extraction and fine-tuning strategies were implemented to enhance learning from the CIFAR datasets.

## Results
The custom ResNet34 model achieved high accuracy on the CIFAR-10 dataset, with promising results on CIFAR-100 as well. The exact figures are documented separately in the training logs.

## Runtime Performance
The table below summarizes the runtime performance in terms of average inference time for the base, traced, and scripted models:

| Model Type    | Average Inference Time (seconds) |
|---------------|----------------------------------|
| Base Model    | 0.032856                         |
| Traced Model  | 0.024855                         |
| Scripted Model| 0.026931                         |

The traced and scripted models, processed via PyTorch's JIT compiler, showed improved inference times compared to the base model, indicating the effectiveness of model optimization techniques.

## Expected Points
Based on the project criteria and the results achieved, the expected number of points is [9 out of 15], considering the model's performance, the optimization techniques applied, and the comprehensive documentation provided.

## Further Work
- Explore additional data augmentation techniques (such as CutMix and MixUp) for CIFAR-100 to improve model accuracy.
- Investigate the impact of different learning rate schedules and optimizer configurations.
- Conduct ablation studies to understand the contribution of each layer and block to the model's performance.
