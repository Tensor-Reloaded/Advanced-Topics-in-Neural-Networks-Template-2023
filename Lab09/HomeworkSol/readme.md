# Image Transformation Benchmarking: CPU vs. GPU

## Overview
This homework involved developing a model to transform CIFAR-10 images (3x32x32) into 1x28x28 grayscale, horizontally and vertically flipped images, focusing on faster processing than sequential CPU transformations. The goal is to demonstrate the model's efficiency on both CPU and GPU, emphasizing batch processing.

## Key Components
1. **Model Architecture**: `TransformationModel` - a compact model incorporating a convolutional layer for minimal learning, followed by deterministic transformations.
2. **Loss Function**: Mean Squared Error (MSE) - chosen for its effectiveness in quantifying the difference between the model's output and ground truth transformations.
3. **Early Stopping**: Implemented to prevent overfitting and ensure efficient training.

[**Wandb Runs**](https://wandb.ai/marius-workspace/ImageTransformationBenchmark?workspace=user-mariusmarin)

[Colab code](https://colab.research.google.com/drive/1OFhPqobpy1sZPRsope1UtYlW9R2Ikbcy#scrollTo=N-Zxi4gQK7xK) (for inference test)

## Script Descriptions
- `main.py`: Sets up the environment, loads data, trains the model, and saves the weights.
- `model.py`: Defines the `TransformationModel`, a neural network for image transformation.
- `utils.py`: Contains utility functions like image generation, comparison, early stopping and timed decorators for performance measurement.
- `data.py`: Handles loading and preprocessing of the CIFAR-10 dataset.
- `train.py`: Manages the training process, including loss computation and early stopping.
- `inference.py`: Used for benchmarking the trained model with various batch sizes and generating comparison images.

Model weights can be found [here](https://drive.google.com/drive/folders/1eYF5ZLfQy1W_saSdh_X-9RgdF59RF8iq).

## Benchmarks
The model demonstrated significantly faster processing on both CPU and GPU compared to sequential transformations, especially notable with batch processing on GPU.

## Image Comparison
Post-training, the model accurately replicated transformations, as can be seen by the sample of images in the LaTeX document. Antialiasing and interpolation were employed to enhance the visual quality of resized images.

## Score Breakdown
- Model creation and transformation capabilities: 3 points
- Loss function utilization and rationale: 2 points
- Implementation of early stopping: 2 points
- Image comparison and documentation: 1 point
- Benchmarking across different devices and parameters: 2 points

Total Expected Points: 10


