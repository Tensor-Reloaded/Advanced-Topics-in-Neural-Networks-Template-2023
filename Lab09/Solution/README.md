### Solution

Neural network for image transformation: 
- building and optimizing a neural network that can perform a combination of Pytorch transforms: resize, flip and grayscale an image.

Model architecture and loss function: 
- the model has a simple architecture with an input layer of size 3 × 32 × 32 and an output layer of size 1 × 28 × 28. 
- the loss function used is Mean Squared Error (MSE), which measures the pixel-wise differences between the generated and target images.

Training process and results: 
- the model is trained on the CIFAR10 dataset and uses Early Stopping to prevent overfitting. 
- the results show that the model outperforms the classical Pytorch transforms in terms of time complexity and produces satisfactory outputs that match the desired transformations.

`runs` -> Tensorboard logs
`image_` -> original image
`label_` -> true transformed
`output_` -> transformed by model
`ImageTransform.pt` -> saved model
`Neural_Network_for_Image_Transformation` -> Latex document
`main.py` -> solution

The expected number of points:

- task 1: complete 3p
- task 2: complete 2p
- task 3: complete 2p
- task 4: complete 1p
- task 5: complete 2p