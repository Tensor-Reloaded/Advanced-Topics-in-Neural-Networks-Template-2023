## Lab5 Homework

I used a model with two hidden layers. The parameters used, the norm and the values ​​of the losses and the accuracies were logged in TensorBoard, some of them were also used in the weights & biases.

We created different configurations to be tested using SGD, Adam, RMSProp, AdaGrad and SGD with SAM optimizers.

The maximum accuracies were achieved using SGD with weight decay and momentum, reaching 46% accuracy on the validation set.
![SGD validation accuracy](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/assets/79158769/64424194-770e-4810-830b-a696b6c93ce4)

In the folder [runs](https://github.com/PopescuAndreiGeorge/Advanced-Topics-in-Neural-Networks-Template-2023/tree/main/Lab05/Solution/Lab5Homework/runs) you can find the logs from TensorBoard, and the projects from weights & biases can be found [here](https://wandb.ai/advanced-topics-in-neural-networks/projects).

Number of points expected: 6
