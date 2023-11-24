Using the pipeline from the previous homework,i simply changed the model.
I implemented the Conv2D layer in a similar manner with nn.Conv2d() and attempted to create a model and see its 
accuracy.
I used RandAugment and Dropout in order to avoid over-fitting.For this reason(and because i noticed that the val
accuracy sometimes declines in an epoch and rises again) i used no_epochs=100 to see if it stabilizes(it did).
The model use is inspired from Nin(Network in Network) using blocks and Pointwise Convolution Layers,with no
residual connections.

Used as optimizer Adam with CrossEntropy and Softmax.

Results:
Highest accuracy: ~74% on CIFAR-10

Expected Points: 4 (due to having only a HandMadeConv2d done)

