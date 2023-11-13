## Lab 

***
Weights and biases: [Sweep](https://wandb.ai/serban-doncean-team/CIFAR10-low?workspace=user-serban-doncean).

Tensorflow logs: [Logs](./runs).

***
Final code: [main](./main.py).

Code used for WandB sweep: [sweep](./sweep.py).

***
Graphs for validation accuracy: [Graphs](./Graphs) 

***
Hyperparameters for each experiment: [Hyperparameters](./Graphs/Hyperparameters.png)

***
I have done several experiments with different models and hyperparameters. 

I have logged the requiered information in Tensorflow log files.

I used the sweep function in Weights and Biases to test randomly created configurations.

I did three or more experiments using the requiered optimizers + AdamW.

I modified the code to use One-Hot Encode. 

I normalized the input data.

I experimented with various augmentations from the Torchvision library.

I used Dropout and a Scheduler to increase my results.

In the end, the best model we obtained using a Dropout with probability 0.3 and the Leaky-ReLu activation function, the maximum validation accuracy on the validation set being 0.647 : [Best run](./Graphs/Best_validation_acc.png).

I expect 7 out of 7 points because I completed all the requierements plus 1 point bonus because I achieved a score better than 0.6 on the validation set.



***

