# Lab5 Apostu Daniel

## Descriere Sweep
Am rulat un sweep cu count 100 pentru a optimiza urmatorii parametri:  
  batch_size: values: [64, 128, 256]  
  learning_rate: values: [0.001, 0.005, 0.01, 0.05, 0.1]  
  optimizer: values: ["SGD", "SGD_SAM", "Adam", "RMSProp", "AdaGrad"]  
  epochs: values: [30, 50, 70]  
  Acestia si ceilalti parametri folositi pot fi vazuti in fisierul sweep_config.yaml.  
  Ca transformari pe langa cele din assignment am folosit v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)  
Link catre sweep: https://wandb.ai/apostudanbs/atnn-lab5/sweeps/pvt2zanv?workspace=user-apostudanbs
## Paneluri care arata importanta parametrilor in sweep:
https://api.wandb.ai/links/apostudanbs/6berpqxb
https://api.wandb.ai/links/apostudanbs/8bdwjdit
https://wandb.ai/apostudanbs/atnn-lab5/reports/undefined-23-11-14-10-09-23---Vmlldzo1OTYzNjQ2?accessToken=urfxu6hwpk9ki4nw6c8qb6qwmul19w94agaeut3yecg32ksgc80ef0zzngxlrbf8

## Cel mai bun run din sweep
Raport: https://api.wandb.ai/links/apostudanbs/k36xs36s

## Cel mai bun run testat:
Separat de sweep, am testat un run cu SGD, 70 de epoci, 0.01 learning rate, batch_size 256, care foloseste nesterov cu momentum de 0.9,
si am ajuns la acuratete la validare de 44.27%.  
Modificare pe care am fost ca la transformari in loc de v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10) sa folosim doar v2.RandomHorizontalFlip().  
Link: https://api.wandb.ai/links/apostudanbs/e0k7u870