import torch
from torch import nn
import matplotlib.pyplot as plt
import time

class Model(nn.Module):
    def __init__(self, image_size, hidden_size, device):
        super(Model, self).__init__()
        self.device = device
        months_input_size = 1
        self.image_layer = nn.Linear(image_size, hidden_size).to(device)
        self.months_layer = nn.Linear(months_input_size, hidden_size).to(device)
        self.output_layer = nn.Linear(hidden_size * 2, image_size).to(device)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, image_input, months_input):
        image_input = image_input.to(self.device)
        months_input = months_input.to(self.device)

        image_output = self.activation(self.image_layer(image_input))
        months_output = self.activation(self.months_layer(months_input))
        combined_output = torch.cat((image_output, months_output), dim=1)
        output = self.output_layer(combined_output)
        return output

    def run(self, train_loader, val_loader, criterion, optimizer, n_epochs):

        train_losses = []
        val_losses = []
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_start_time = time.time()

            train_loss = self.train_model(train_loader, criterion, optimizer)
            val_loss = self.validate_model(val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"Epoch: {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds")

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Total Time: {total_duration:.2f} seconds")

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def train_model(self, train_loader, criterion, optimizer):
        self.train()
        running_loss = 0.0
        for images_start, images_end, months_between in train_loader:

            images_start = images_start.to(self.device)
            images_end = images_end.to(self.device)
            months_between = months_between.to(self.device)

            optimizer.zero_grad()
            outputs = self(images_start, months_between)
            loss = criterion(outputs, images_end)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images_start.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate_model(self, validation_loader, criterion):
        self.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images_start, images_end, months_between in validation_loader:

                images_start = images_start.to(self.device)
                images_end = images_end.to(self.device)
                months_between = months_between.to(self.device)

                outputs = self(images_start, months_between)
                loss = criterion(outputs, images_end)
                running_loss += loss.item() * images_start.size(0)
        epoch_loss = running_loss / len(validation_loader.dataset)
        return epoch_loss