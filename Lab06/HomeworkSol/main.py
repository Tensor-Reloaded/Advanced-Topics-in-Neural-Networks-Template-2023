import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from config import config
from data_utils import prepare_data_loaders
from model_utils import initialize_resnet
from train_utils import train_one_epoch, validate, test


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    wandb.init(project="deep_learning_project", config=config)

    num_classes = 10 if config['dataset_name'] == 'CIFAR10' else 100

    trainloader, validloader, testloader = prepare_data_loaders(config)

    model = initialize_resnet(config['model_name'], num_classes,
                              use_pretrained=config['use_pretrained'],
                              feature_extract=config['feature_extract'])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_val_accuracy = 0.0

    for epoch in range(config['num_epochs']):
        train_loss, train_accuracy = train_one_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, validloader, criterion, device)
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "lr": scheduler.get_last_lr()[0]
        })

        # Output the metrics for this epoch
        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.to('cpu').model.state_dict(), f"best_model_epoch_{epoch}.pth")
            wandb.save(f"best_model_epoch_{epoch+1}.pth")
            print(f"New best model saved at epoch {epoch + 1} with Validation Accuracy: {val_accuracy}%")

    test_accuracy = test(model, testloader, device)
    print(f"Test Accuracy: {test_accuracy}%")
    wandb.log({'test_accuracy': test_accuracy})

    wandb.finish()


if __name__ == '__main__':
    main()
