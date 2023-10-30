import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, image_size, hidden_size):
        super(Model, self).__init__()
        months_input_size = 1
        self.image_layer = nn.Linear(image_size, hidden_size)
        self.months_layer = nn.Linear(months_input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size * 2, image_size)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, image_input, months_input):
        image_output = self.leakyrelu(self.image_layer(image_input))
        months_output = self.leakyrelu(self.months_layer(months_input))
        combined_output = torch.cat((image_output, months_output), dim=1)
        output = self.output_layer(combined_output)
        return output

    def run(self, train_loader, val_loader, criterion, optimizer, n_epochs):
        for epoch in range(n_epochs):
            train_loss = self.train_model(train_loader, criterion, optimizer)
            val_loss = self.validate_model(val_loader, criterion)
            print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def train_model(self, train_loader, criterion, optimizer):
        self.train()
        running_loss = 0.0
        for images_start, images_end, months_between in train_loader:
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
                outputs = self(images_start, months_between)
                loss = criterion(outputs, images_end)
                running_loss += loss.item() * images_start.size(0)
        epoch_loss = running_loss / len(validation_loader.dataset)
        return epoch_loss
