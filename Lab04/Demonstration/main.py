import torch
import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from datasets import MyDataset
from models import MLP
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    complete_dataset = MyDataset('Homework Dataset')
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, [0.7, 0.15, 0.15])
    #to cpu
    model = MLP(inputDimensions=49153, outputDimensions=49152)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    for epoch in range(10):
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            output = model(batch[0])
            loss = criterion(output, batch[1])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(validation_dataloader):
                output = model(batch[0])
                loss = criterion(output, batch[1])
                print(loss.item())
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            output = model(batch[0])
            loss = criterion(output, batch[1])
            print(loss.item())

    #to gpu
    # model = MLP(inputDimensions=49153, outputDimensions=49152)
    # model.load_state_dict(torch.load('model.pth'))
    # model.cuda()
    # criterion = nn.L1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    # for epoch in range(10):
    #     model.train()
    #     for batch in tqdm(train_dataloader):
    #         optimizer.zero_grad()
    #         output = model(batch[0].cuda())
    #         loss = criterion(output, batch[1].cuda())
    #         loss.backward()
    #         optimizer.step()
    #     model.eval()
    #     with torch.no_grad():
    #         for batch in tqdm(validation_dataloader):
    #             output = model(batch[0].cuda())
    #             loss = criterion(output, batch[1].cuda())
    #             print(loss.item())
    # model.eval()
    # with torch.no_grad():
    #     for batch in tqdm(test_dataloader):
    #         output = model(batch[0].cuda())
    #         loss = criterion(output, batch[1].cuda())
    #         print(loss.item())
