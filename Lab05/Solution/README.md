<a target="_blank" href="https://colab.research.google.com/github/Nick24-hub/Advanced-Topics-in-Neural-Networks-Template-2023/blob/main/Lab05/Solution/lab5_assignment.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- implemented logging system using Tensorboard
- evaluated the performance of SGD, Adam, RMSprop, AdaGrad
- best accuracy obtained: 0.4823 on validation and 0.4228 on train with Adam with learning_rate = 0.001 and batch size = 256 after 100 epochs
- obtained better accuracy on validation because I used v2.RandAugment() in trainset trainsform pipeline

Expected points: 5