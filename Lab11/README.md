## Lab 11
***
## RNNs

Check:
 - RNN tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

***
## Optimizing PyTorch pipelines

"Another benefit of striving for efficiency is that the process forces you to understand the problem in more depth." - Alex Stepanov
***
Caution: Spending 5 hours to optimize a single pipeline that runs for only 1 hour is a waste of time.

Goals:
 - Optimize by default. Write a training and inference pipeline that is reasonably fast, from the beginning.
 - Do not do premature optimization.
 - Find a balance between accuracy and performance, depending on your problem.
 - Find a balance between the time spent for optimizing and the time spent for doing experiments.

Resources: 
- https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html 
- https://paulbridger.com/posts/pytorch-tuning-tips/ - Good resource, but conclusions are a bit old (April 2023).
- https://paulbridger.com/posts/mastering-torchscript/ - Good resource for scripting the model.
- https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html - Automatic Mixed Precision.

