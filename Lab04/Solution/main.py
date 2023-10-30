import torch
import torch.nn as nn
import torch.optim as optim
from dataset import *
from model import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def run(n,cfg,dataloader_train,dataloader_val):
    var = tqdm(range(n))
    train_losses=[]
    val_losses=[]
    cfg['model'].to(cfg['device'])
    for epoch in var:
        train_loss=train(cfg,dataloader_train,var)
        train_losses.append(train_loss)
        val_loss=val(config,dataloader_val)
        val_losses.append(val_loss)
        var.set_description(f"train_loss:{train_loss}, val_loss: {val_loss}")
    return train_losses,val_losses
def train(cfg,dataloader_train,var):
    total_loss=0
    step=0
    cfg['model'].train()
    for img,label,time in dataloader_train:
        cfg['optimizer'].zero_grad()
        img,label,time=img.to(cfg['device']),label.to(cfg['device']),time.to(cfg['device'])
        img = img.to(torch.float32)
        label = label.to(torch.float32)
        output=cfg['model'](img,time)
        loss=cfg['criterion'](output,label)
        total_loss+=loss.item()
        loss.backward()
        step+=1
        cfg['optimizer'].step()
        #var.set_description(f"{step}/, train_loss: {loss.item()}")
    total_loss/=step
    return total_loss
    
def val(cfg,dataloader_val):
    total_loss=0
    step=0
    cfg['model'].eval()
    for img,label,time in dataloader_val:
        img,label,time=img.to(cfg['device']),label.to(cfg['device']),time.to(cfg['device'])
        img = img.to(torch.float32)
        label = label.to(torch.float32)
        output=cfg['model'](img,time)
        loss=cfg['criterion'](output,label)
        step+=1
        total_loss+=loss.item()
    return total_loss/step
    

    
img_dim=128
transforms_train = A.Compose([
    A.Resize(img_dim,img_dim),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=90,p=0.5),
    ToTensorV2()
],
    additional_targets={'label': 'image'}
)


# Generate train test split indices
total_dataset = LandscapeDataset("./Homework Dataset",transforms=transforms_train)

print("Number of samples", len(total_dataset))

img_in,img_out,time=total_dataset[0]
print(img_out.shape[0])
print(img_in.shape)

train_dataset,val_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.15,0.15])

config={}
config['batch_size']=64
config['num_workers']=2
config['model']=SimpleModel(img_in.shape[0],img_out.shape[0])
config['learning_rate']=0.05
config['weight_decay']=0.005
config['criterion']=torch.nn.CrossEntropyLoss()
config['optimizer']=optim.AdamW(config['model'].parameters(),lr=config['learning_rate'], weight_decay=config['weight_decay'])
config['device']=torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

dataloader_train = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'],
                             drop_last=True)
dataloader_val = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
dataloader_test = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

train_losses,val_losses=run(10,config,dataloader_train,dataloader_val)



plt.plot(train_losses)
plt.savefig('./Train_loss.png')
plt.clf()
plt.plot(val_losses)
plt.savefig('./Val_loss.png')
