from datasets import materialize_wrapper, ImageComparisonDatasetWrapper
from dataset_load_logic import load_data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from neural_network import SatelliteConv
from torch.utils.data import Subset
from torchvision import transforms
from torch import nn
from torch import optim
from datasets import same_transform
from trainer import run
import seaborn as sns


def display_images(dataset: ImageComparisonDatasetWrapper):
    print(f"Dataset len: {len(dataset)}")
    lh_img, rh_img, days_between = dataset[0]
    plt.imshow(lh_img.permute(1, 2, 0))
    plt.show()
    plt.imshow(rh_img.permute(1, 2, 0))
    plt.show()


def visualize(dataset: Subset, model: SatelliteConv):
    in_img, out_img, days_between = dataset[len(dataset) // 2]

    plt.imshow(in_img.permute(1, 2, 0))
    plt.show()
    plt.imshow(out_img.permute(1, 2, 0))
    plt.show()

    plt.imshow(model((in_img, days_between)).permute(1, 2, 0).detach().numpy())
    plt.show()


def plot_evolution(evolution: list, title: str):
    sns.lineplot(data=evolution).set_title(title)
    plt.show()


if __name__ == "__main__":
    SatelliteImagesDataset = materialize_wrapper(load_data)
    satellite_dataset = SatelliteImagesDataset(
        r"D:\personal\CARN\Advanced-Topics-in-Neural-Networks-2023\Lab04\Homework Dataset",
        [same_transform(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))),
         same_transform(transforms.RandomErasing(p=0.2, scale=(0.01, 0.01), value=1)),
         same_transform(transforms.RandomHorizontalFlip(p=0.5)),
         same_transform(transforms.Resize((64, 64)))],
        random_rotation=(0, 20)
    )

    display_images(satellite_dataset)

    train_subset, val_subset, test_subset = satellite_dataset.split(0.7, 0.15, 0.15)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    satellite_model = SatelliteConv(image_dims=(3, 64, 64))
    model_loss_fn = nn.MSELoss()
    model_optimizer = optim.Adam(satellite_model.parameters(), lr=0.001)

    visualize(test_subset, satellite_model)
    train_evo, val_evo = run(2, satellite_model, train_subset, val_subset, model_loss_fn, model_optimizer)
    visualize(test_subset, satellite_model)

    plot_evolution(train_evo, "Train evolution")
    plot_evolution(val_evo, "Val evolution")
