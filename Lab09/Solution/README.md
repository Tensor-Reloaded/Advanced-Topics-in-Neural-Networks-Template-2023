### Homework 7

Neural Network architecture for Image Transformation:

This project involves constructing a neural network capable of executing a composite of PyTorch transforms(resize, flip, and grayscale operations) on an image.
The neural network features a straightforward architecture, comprising an input layer with dimensions 3 × 32 × 32 and an output layer with dimensions 1 × 28 × 28.

Loss Function Selection:

The chosen loss function is Mean Squared Error (MSE), quantifying the pixel-wise disparities between the generated and target images.

Training and Performance:

The model undergoes training using the CIFAR10 dataset, incorporating Early Stopping mechanisms to curb overfitting tendencies. Evaluation results indicate that the model surpasses conventional PyTorch transforms not only in terms of computational efficiency but also in generating outputs that closely align with the desired image transformations.


Wandb logs: https://wandb.ai/brezuleanu-mihai-alexandru/Homework7/runs/nqychpz7?workspace=user-brezuleanu-mihai-alexandru

The expected number of points is 10.
