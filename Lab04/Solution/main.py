from datasets import materialize_wrapper, ImageComparisonDatasetWrapper
from dataset_load_logic import load_data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from neural_network import SatelliteConv
from torch import nn
from torch import optim
from torch.utils.data import Subset
from tqdm import tqdm


def display_images(dataset: ImageComparisonDatasetWrapper):
    print(f"Dataset len: {len(dataset)}")
    lh_img, rh_img, days_between = dataset[0]
    plt.imshow(lh_img.permute(1, 2, 0))
    plt.show()
    plt.imshow(rh_img.permute(1, 2, 0))
    plt.show()


def test(model, dataset: Subset, loss_fn, optimizer):
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(total=len(dataset), desc="Training", dynamic_ncols=True)

        for in_img, out_img, nr_days in dataset:
            optimizer.zero_grad()
            outputs = model((in_img, nr_days))
            loss = loss_fn(outputs, out_img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

        pbar.close()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataset)}\n')


if __name__ == "__main__":
    SatelliteImagesDataset = materialize_wrapper(load_data)
    satellite_dataset = SatelliteImagesDataset(
        r"D:\personal\CARN\Advanced-Topics-in-Neural-Networks-2023\Lab04\Homework Dataset", None,
        random_rotation=(0, 20)
    )

    display_images(satellite_dataset)

    train_subset, val_subset, test_subset = satellite_dataset.split(0.7, 0.15, 0.15)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    satellite_model = SatelliteConv(image_dims=(3, 128, 128))
    model_loss_fn = nn.MSELoss()
    model_optimizer = optim.Adam(satellite_model.parameters(), lr=0.001)

    test(satellite_model, train_subset, model_loss_fn, model_optimizer)
