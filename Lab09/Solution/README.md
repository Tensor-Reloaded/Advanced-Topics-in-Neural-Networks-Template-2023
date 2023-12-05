## Assignment 7

Created the [model](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/blob/main/Lab09/Solution/Assignment7/model.py) to apply the required transformations.

Tested different loss functions for training and found the one that best suited the model.

Created a class that deals with [early stopping](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/blob/main/Lab09/Solution/Assignment7/early_stopping.py) based on the difference in loss values from the last epochs.

Explanations and examples of all the mentioned can be found in [Assignment 7.pdf](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/blob/main/Lab09/Solution/Assignment%207.pdf).

Defined a [function to load the weights](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/blob/main/Lab09/Solution/Assignment7/inference.py) of the model and run the "test inference time" function. The results obtained can be seen below. Image transformations on the GPU are performed faster than the real transformations.

The logs from TensorBoard can be found in the [runs](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/tree/main/Lab09/Solution/Assignment7/runs) file, those from Weights & Biases in the [wandb](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/tree/main/Lab09/Solution/Assignment7/wandb) file the projects from Weights & Biases can be found [here](https://wandb.ai/advanced-topics-in-neural-networks/projects).

![image (2)](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/assets/79158769/78c0a9d1-8f77-4a2a-b233-8159b47d2126)
![image (3)](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/assets/79158769/08fe8df0-c235-4200-9556-0f7586110ff3)

The model I trained for testing the execution times of the transformations can be found in the [trainedModels](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/tree/main/Lab09/Solution/Assignment7/trainedModels) file.

Expected number of points: 9
