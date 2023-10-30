
from datasets import AgedCityImagesDataset

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from model import CityLearner

data_folder = "C:\\Users\\Gabriel\\Desktop\\Laborator\\ATNN\\Advanced-Topics-in-Neural-Networks-Template-2023\\Lab4_me\\dataset"



def print_hi(name):

    dataset = AgedCityImagesDataset(data_folder)

    #[start_image, end_image, time_skip] = dataset[26]

    total_datapoints = len(dataset)
    train_size = int(0.7 * total_datapoints)
    val_size = int(0.15 * total_datapoints)

    # Create lists of indices for train, validation, and test sets
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(dataset)))

    # Initialize lists for train, validation, and test files
    train_files = []
    val_files = []
    test_files = []

    # Fetch individual elements for train set
    for idx in train_indices:
        image1, image2, time_skip = dataset[idx]
        train_files.append((image1, image2, time_skip))

    # Fetch individual elements for validation set
    for idx in val_indices:
        image1, image2, time_skip = dataset[idx]
        val_files.append((image1, image2, time_skip))

    # Fetch individual elements for test set
    for idx in test_indices:
        image1, image2, time_skip = dataset[idx]
        test_files.append((image1, image2, time_skip))

    batch_size = 32
    train_loader = DataLoader(train_files, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_files, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_files, batch_size=batch_size, shuffle=False)

    # Instantiate your custom model
    model = CityLearner()  # Define your model class
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for image_pair, duration, target_image in train_loader:
            optimizer.zero_grad()
            output_image = model(image_pair, duration)
            loss = criterion(output_image, target_image)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

    # Evaluation
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for image_pair, duration, target_image in test_loader:
            output_image = model(image_pair, duration)
            loss = criterion(output_image, target_image)
            test_loss += loss.item()

    print(f'Test Loss: {test_loss / len(test_loader)}')


    #image1 = start_image.numpy()
    #image2 = end_image.numpy()
   # cv2.imshow("Image 1", image1)
   # cv2.imshow("Image 2", image2)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    print_hi('PyCharm')

