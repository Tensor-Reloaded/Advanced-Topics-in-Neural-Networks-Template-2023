import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from datasets import MetaDataset
from model import Model

from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

def main():
    transform = transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.Lambda(lambda x: x.view(-1)) # flatten
    ])

    dataset = MetaDataset('C:/School/Sem1/CARN/Advanced-Topics-in-Neural-Networks-Template-2023/Lab04/Homework Dataset/', transform)

    train_size = int(0.7 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(
        image_input_size=16384, 
        months_between_input_size=1, 
        hidden_size=128, 
        device=device
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    n_epochs = 300 
    model.run(train_loader, validation_loader, criterion, optimizer, n_epochs)

    test_loss = model.evaluate_model(test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')

    image_path = 'C:/School/Sem1/CARN/Advanced-Topics-in-Neural-Networks-Template-2023/Lab04/Homework Dataset/L15-0331E-1257N_1327_3160_13/images/global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13.tif'

    img = Image.open(image_path)
    img = transform(img)
    img = img.to(device).unsqueeze(0)

    months = 6 
    months = torch.tensor([months], dtype=torch.float32).unsqueeze(0)
    months = months.to(device)

    with torch.no_grad():
        output = model.forward(img, months)

    output = output.view(1, int(output.size(-1) ** 0.5), -1)  # Reshape
    output = output * 0.5 + 0.5  # Denormalize

    plt.imshow(output.squeeze().cpu().numpy(), cmap='gray')
    plt.show()

if(__name__ == "__main__"):
    main()