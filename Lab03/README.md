## Lab 3

***
Lab Notebook: [TensorOperations.ipynb](./TensorOperations.ipynb)

***
Lab Assignment: MLP implementation on MNIST, with shape: `[784, 100, 10]` (Deadline: PR by End-of-Day Monday, 23.10.2023)

***
For self-study:
* [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (animated introduction to neural networks and backpropagation)
* [Essence of calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (derivatives; chain rule)

Advanced:
* TorchScript (PyTorch jit): https://pytorch.org/docs/stable/jit.html
* PyTorch jit trace: https://pytorch.org/docs/stable/generated/torch.jit.trace.html
* PyTorch jit script: https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script
* Pytorch compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
  * `torch.compile` does not work on Windows systems!
  * Always measure whether compiling your model improves the performance or not! 
  
***
Pleas check the Lab 2 Assignment's solution: (will be uploaded after Lab 3).

***
References:
 - MNIST: https://pytorch.org/vision/0.15/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
 - PyTorch Benchmarking: https://pytorch.org/tutorials/recipes/recipes/benchmark.html