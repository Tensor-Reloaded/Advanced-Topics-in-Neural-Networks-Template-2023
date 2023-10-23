import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F


# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Forward propagation
def forward(x):
    x = x.view(x.size(0), -1)
    x1 = torch.mm(x, w1) + b1
    x1 = sigmoid(x1)
    x2 = torch.mm(x1, w2) + b2
    return x2

# Training
def train(train_loader, w1, b1, w2, b2, learning_rate):
    total_loss = 0
    correct = 0
    for data, target in train_loader:
        output = forward(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()

        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            b1 -= learning_rate * b1.grad
            w2 -= learning_rate * w2.grad
            b2 -= learning_rate * b2.grad

        predicted = output.argmax(dim=1)
        correct += predicted.eq(target).sum().item()

    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


# Validation
def validate(val_loader):
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = forward(data)
            total_loss += F.cross_entropy(output, target).item()
            predicted = output.argmax(dim=1)
            correct += predicted.eq(target).sum().item()
    return total_loss / len(val_loader.dataset), correct / len(val_loader.dataset)


input_size = 784
hidden_size = 100
output_size = 10

w1 = torch.randn(input_size, hidden_size, requires_grad=True)
b1 = torch.zeros(hidden_size, requires_grad=True)
w2 = torch.randn(hidden_size, output_size, requires_grad=True)
b2 = torch.zeros(output_size, requires_grad=True)

learning_rate = 0.001
batch_size = 64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(train_loader, w1, b1, w2, b2, learning_rate)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy*100:.2f}%")

val_loss, val_accuracy = validate(val_loader)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")