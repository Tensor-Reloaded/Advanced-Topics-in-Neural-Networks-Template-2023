import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import *
from tqdm import tqdm

from models import ImageMLP
import matplotlib.pyplot as plt


total_dataset = ImageDataset(dataset_path="Homework Dataset")
print(len(total_dataset))
aux, aux2 = total_dataset[0]
print(aux.shape, aux2.shape)

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.15, 0.15])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model, loss function and optimizer
model = ImageMLP(input_dim=aux.shape[0], \
                       output_dim=aux2.shape[0], \
                       output_activation=nn.Softmax(dim=1))

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

device = 'cpu'
num_epochs = 100


def train():
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        total += labels.size(0)
        correct += (outputs == labels).sum().item()
    train_acc = 100 * correct / total
    return train_acc, total_loss

def validate():
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in validation_loader:
            outputs = model(features)
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
    validate_acc = 100 * correct / total
    return validate_acc


def run(num_epochs):
    pbar = tqdm(range(num_epochs))
    train_accs = []
    validate_accs = []
    for epoch in pbar:
        train_acc, total_loss = train()
        validate_acc = validate()
        pbar.set_postfix_str(
        f'Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_acc}%, Train Loss: {total_loss / len(train_loader)}, Validation Accuracy: {validate_acc}%\n')
        train_accs.append(train_acc)
        validate_accs.append(validate_acc)

    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), train_accs, label='Train Accuracy', color='blue')
    ax.plot(range(num_epochs), validate_accs, label='Validate Accuracy', color='red')
    plt.show()



run(num_epochs)

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
     for features, labels in test_loader:
        outputs = model(features)
        total += labels.size(0)
        correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

print(100 * correct / total)