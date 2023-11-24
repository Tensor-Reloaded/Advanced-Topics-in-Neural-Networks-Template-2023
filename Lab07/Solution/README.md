## Lab6 Homework

I created a class in which I did the [manual implementation of Conv2d](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/blob/main/Lab07/Solution/Lab6Homework/handmadeConv2d.py) and at the beginning of the main function I ran a test by comparing the result obtained with the one returned by the torch.nn function.

I created [a model](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/blob/main/Lab07/Solution/Lab6Homework/denseNetModel.py) based on Densely Connected Convolutional Networks, more precisely, based on DenseNet-121. I tried to follow the structure present in the pytorch implementation for the construction of the model and I tested it by changing the parameters for learning, but also of the model, such as the number of feature maps produced for each layer (growth rate).

In the folder [runs](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/tree/main/Lab07/Solution/Lab6Homework/runs) you can find the logs from TensorBoard, and the projects from weights & biases can be found [here](https://wandb.ai/advanced-topics-in-neural-networks/projects) (the ones starting with "Homework6" are of interest for this assignment).

Number of points expected: 4
