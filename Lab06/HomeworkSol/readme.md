# Project Summary - Custom ResNet34 for CIFAR-10 and CIFAR-100

## Overview
This project involved implementing a custom ResNet34 architecture suitable for classifying images from the CIFAR-10 and CIFAR-100 datasets. It includes a detailed examination of the architecture, a focus on transfer learning, and performance optimizations using PyTorch's JIT compilation.

## Implementation Details
- Custom ResNet34 model was developed to closely follow the architecture of torchvision models.
- Transfer learning was employed by initializing the custom model with weights from the pretrained torchvision ResNet34 model.
- The model was adapted for CIFAR datasets by adjusting the final fully connected layer to match the number of target classes.
- Feature extraction and fine-tuning strategies were implemented to enhance learning from the CIFAR datasets.

## Script Summaries
### main.py
- **Purpose**: Core training script for the project.
- **Functionalities**: Initializes and configures the model, data loaders, and training parameters. Manages the training loop, saves the best model, logs progress, and tests the final model.

### model_utils.py
- **Purpose**: Defines the custom ResNet model and initialization functions.
- **Functionalities**: Implements the ResNet34 and BasicBlock classes and provides functions for model initialization with options for pretrained weights and feature extraction.

### data_utils.py
- **Purpose**: Manages data loading and preprocessing.
- **Functionalities**: Implements image augmentation, loads and preprocesses CIFAR datasets, and handles data loaders.

### train_utils.py
- **Purpose**: Contains utility functions for training and validation.
- **Functionalities**: Manages training, validation, and testing phases, including forward pass, loss computation, and optimization.

### config.py
- **Purpose**: Central configuration file for the project.
- **Functionalities**: Holds configurable parameters for easy modification of training settings.

### inference.py
- **Purpose**: Script for model inference and validation accuracy calculation.
- **Functionalities**: Loads the best saved model and runs inference on the validation dataset to print accuracy.

### handmade_conv.py
- **Purpose**: Implements a custom convolutional layer from scratch.
- **Functionalities**: Demonstrates the mechanics of a convolutional layer, provides testing and comparison with PyTorch’s built-in layers and produces the same results to within a small tolerance.

### trace_and_script.py
- **Purpose**: Demonstrates the use of PyTorch JIT compilation for model optimization.
- **Functionalities**:
  - Measures the average inference time of the base, traced, and scripted models.
  - Utilizes `torch.jit.trace` and `torch.jit.script` to create optimized versions of the ResNet34 model.
  - Provides a clear comparison of the performance improvement from JIT compilation.

## Results
The custom ResNet34 model achieved 94.1% validation accuracy on the CIFAR-10 dataset, with promising results on CIFAR-100 as well. The evolution of validation accuracy is shown below:

![CIFAR-10 validation accuracy metric](https://github.com/mariusmarin98/Advanced-Topics-in-Neural-Networks-Template-2023/blob/main/Lab06/HomeworkSol/cifar10-validation-accuracy.png)

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|-------|------------|---------------|----------|-------------|
| 1     | 1.8657     | 31.18         | 1.5583   | 42.18       |
| 2     | 1.4469     | 47.29         | 1.3084   | 56.42       |
| 3     | 1.2065     | 56.58         | 1.0383   | 64.64       |
| 4     | 1.0380     | 62.88         | 0.9088   | 70.60       |
| 5     | 0.9029     | 68.06         | 1.2168   | 69.06       |
| 6     | 0.7920     | 72.44         | 0.6722   | 76.84       |
| 7     | 0.7183     | 74.86         | 0.5749   | 80.06       |
| 8     | 0.6558     | 77.22         | 0.5512   | 80.76       |
| 9     | 0.6082     | 78.86         | 0.5352   | 82.02       |
| 10    | 0.5612     | 80.41         | 0.4812   | 83.62       |
| 11    | 0.5330     | 81.62         | 0.4460   | 85.08       |
| 12    | 0.4911     | 82.82         | 0.3863   | 86.40       |
| 13    | 0.4631     | 84.07         | 0.3913   | 86.52       |
| 14    | 0.4462     | 84.69         | 0.4612   | 84.74       |
| 15    | 0.4208     | 85.50         | 0.3549   | 87.40       |
| 16    | 0.4038     | 85.84         | 0.3880   | 86.42       |
| 17    | 0.3828     | 86.63         | 0.3679   | 88.24       |
| 18    | 0.3621     | 87.42         | 0.3499   | 88.00       |
| 19    | 0.3526     | 87.77         | 0.3090   | 89.32       |
| 20    | 0.3370     | 88.27         | 0.3045   | 89.76       |
| 21    | 0.3182     | 88.87         | 0.2904   | 90.26       |
| 22    | 0.3111     | 89.11         | 0.2816   | 90.18       |
| 23    | 0.2991     | 89.59         | 0.3096   | 89.70       |
| 24    | 0.2860     | 89.87         | 0.2913   | 90.52       |
| 25    | 0.2737     | 90.48         | 0.2814   | 90.40       |
| 26    | 0.2647     | 90.70         | 0.2798   | 90.82       |
| 27    | 0.2610     | 90.97         | 0.2554   | 91.40       |
| 28    | 0.2438     | 91.41         | 0.2956   | 90.18       |
| 29    | 0.2362     | 91.70         | 0.2643   | 90.98       |
| 30    | 0.2283     | 92.00         | 0.2560   | 91.70       |
| 31    | 0.1672     | 94.18         | 0.1973   | 93.44       |
| 32    | 0.1485     | 94.90         | 0.1934   | 93.58       |
| 33    | 0.1392     | 95.05         | 0.1986   | 93.70       |
| 34    | 0.1317     | 95.43         | 0.1976   | 93.50       |
| 35    | 0.1281     | 95.50         | 0.1958   | 93.78       |
| 36    | 0.1277     | 95.52         | 0.1979   | 93.72       |
| 37    | 0.1218     | 95.70         | 0.2082   | 93.56       |
| 38    | 0.1193     | 95.80         | 0.1980   | 93.82       |
| 39    | 0.1173     | 95.98         | 0.1991   | 93.82       |
| 40    | 0.1155     | 95.93         | 0.1963   | 93.88       |
| 41    | 0.1132     | 96.04         | 0.2027   | 93.84       |
| 42    | 0.1116     | 96.14         | 0.2044   | 93.78       |
| 43    | 0.1074     | 96.28         | 0.2004   | 93.90       |
| 44    | 0.1062     | 96.28         | 0.2032   | 94.04       |
| 45    | 0.1062     | 96.27         | 0.2076   | 93.88       |
| 46    | 0.1002     | 96.52         | 0.2073   | 93.90       |
| 47    | 0.0974     | 96.63         | 0.2075   | 94.00       |
| 48    | 0.1036     | 96.33         | 0.2030   | 93.96       |
| 49    | 0.0975     | 96.69         | 0.2113   | 93.92       |
| 50    | 0.0959     | 96.65         | 0.2070   | 93.96       |
| 51    | 0.0969     | 96.60         | 0.2093   | 94.12       |
| 52    | 0.0978     | 96.65         | 0.2117   | 94.00       |
| 53    | 0.0916     | 96.92         | 0.2121   | 93.86       |
| 54    | 0.0904     | 96.80         | 0.2105   | 93.88       |
| 55    | 0.0922     | 96.86         | 0.2090   | 93.84       |
| 56    | 0.0866     | 96.99         | 0.2111   | 93.84       |
| 57    | 0.0863     | 96.97         | 0.2122   | 94.26       |
| 58    | 0.0869     | 96.96         | 0.2107   | 94.12       |
| 59    | 0.0849     | 97.06         | 0.2093   | 94.14       |
| 60    | 0.0859     | 97.01         | 0.2145   | 94.10       |

## Model Weights
Model weights, including pretrained weights for transfer learning, can be found [here](https://drive.google.com/drive/folders/1nJTLvwz8noIO7K9NBT2muGiDwjBR2ESf) (ran with CUDA).

## W&B Runs Links
- [CIFAR-10 Classification Run](https://wandb.ai/marius-workspace/cifar10_classification/runs/f3ozgmjj?workspace=user-mariusmarin)
- [Deep Learning Project Workspace](https://wandb.ai/marius-workspace/deep_learning_project?workspace=user-mariusmarin)

## Runtime Performance
The table below summarizes the runtime performance in terms of average inference time for the base, traced, and scripted models:

| Model Type    | Average Inference Time (seconds) |
|---------------|----------------------------------|
| Base Model    | 0.032856                         |
| Traced Model  | 0.024855                         |
| Scripted Model| 0.026931                         |

The traced and scripted models, processed via PyTorch's JIT compiler, showed improved inference times compared to the base model, indicating the effectiveness of model optimization techniques.


## Expected Points breakdown
Based on the project criteria and the results achieved, the expected number of points is [9.5 out of 16], considering the model's performance, the optimization techniques applied, and the comprehensive documentation provided.


| Task Description                                             | Points Earned | Maximum Points |
|--------------------------------------------------------------|---------------|----------------|
| Custom Conv2D Layer implementation and validation            | 4             | 4              |
| Model compilation, tracing, and scripting, check runtime     | 1             | 1              |
| 94% validation accuracy on CIFAR-10 without a pretrained model| 2             | 2              |
| 95.5% validation accuracy on CIFAR-10 without a pretrained model| 0           | 2              |
| Transfer learning on CIFAR-10 to achieve 97% accuracy        | 1             | 2              |
| 97% validation accuracy on CIFAR-10 without a pretrained model| 0            | 2              |
| 75% validation accuracy on CIFAR-100 without a pretrained model| 0.5-1       | 2              |
| Testing a pretrained model on CIFAR-100 and comparing results| 1             | 1              |
| **Total**                                                    | **9.5**        | **16**         |

## Further Work
- Explore additional data augmentation techniques (such as CutMix and MixUp) for CIFAR-100 to improve model accuracy.
- Investigate the impact of different learning rate schedules and optimizer configurations.
- Conduct ablation studies to understand the contribution of each layer and block to the model's performance.
