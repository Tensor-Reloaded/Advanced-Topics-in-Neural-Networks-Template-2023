![image_clipdrop-enhance](https://github.com/Tensor-Reloaded/Advanced-Topics-in-Neural-Networks-Template-2023/assets/8055539/5965f7aa-34ad-4899-b2af-be3cc084cb96)

# Advanced Topics in Neural Networks Template 2023

Repository for the Advanced Topics in Neural Networks laboratory, "Alexandru Ioan Cuza" University, Faculty of Computer Science, Master degree.

## Environment setup

PyTorch, Pandas, Numpy, Tensorboard, Matplotlib, and Opencv are already available in Google Colab.

Local installation: 
1. Create a Python virtual environment (the current stable version for PyTorch 2.0.1 is Python 3.10).
    * If you are using `conda`, use `conda config --add channels conda-forge` first to add `conda-forge` as your highest priority channel.
3. Activate the virtual environment and install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) using `conda` or `pip`, depending on your environment.
    * Choose the Stable Release, choose your OS, select Conda or Pip and your compute platform. For Linux and Windows, CUDA 1X.X or CPU builds are available, while for Mac, only builds with CPU and MPS acceleration.
    * Example CPU: ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```.
4. Install Tensorboard.
    * `conda install -c conda-forge tensorboard` / `pip install tensorboard`.
5. Install Matplotlib.
    * `conda install -c conda-forge matplotlib` / `pip install matplotlib`.
6. Install Opencv.
    * `conda install -c conda-forge opencv` / `pip install opencv-python`.


## Recommended resources:

- Linear algebra:
   * [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (linear transformations; matrix multiplication)
   * [Essence of calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (derivatives; chain rule)
- Backpropagation:
   * [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (animated introduction to neural networks and backpropagation)
- Convolutions:
   * [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) (convolution example; convolutions in image processing; convolutions and polynomial multiplication; FFT)
