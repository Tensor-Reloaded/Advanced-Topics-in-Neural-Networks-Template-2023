# Advanced Topics in Neural Networks Template 2023

Repository for the Advanced Topics in Neural Networks laboratory, "Alexandru Ioan Cuza" University, Faculty of Computer Science, Master degree.

## Environment setup

PyTorch, Pandas, Numpy, Tensorboard, Matplotlib and Opencv are aleady available in Google Colab.

Local installation: 
1. Create a Python virtual environment (current stable version is Python 3.10).
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
