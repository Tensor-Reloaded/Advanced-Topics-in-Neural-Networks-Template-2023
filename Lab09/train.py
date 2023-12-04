import torch
import wandb
from utils import EarlyStopping


def train_model(model, device, train_loader, val_loader, loss_function, optimizer, epochs):
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, ground_truth in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            ground_truth = ground_truth.to(device)
            outputs = model(images)
            loss = loss_function(outputs, ground_truth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        wandb.log({"Train Loss": train_loss})

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ground_truth in val_loader:
                images = images.to(device)
                ground_truth = ground_truth.to(device)
                outputs = model(images)
                val_loss += loss_function(outputs, ground_truth).item()

        val_loss /= len(val_loader)
        wandb.log({"Validation Loss": val_loss})

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    # Save model weights
    torch.save(model.state_dict(), 'model_weights.pth')
    #wandb.save('model_weights.pth')
