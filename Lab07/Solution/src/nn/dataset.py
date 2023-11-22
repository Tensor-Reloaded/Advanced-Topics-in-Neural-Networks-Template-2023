import torch.utils.data as torch_data


class CachedDataset(torch_data.Dataset):
    dataset: any

    def __init__(
        self,
        dataset: any,
        cache: bool,
    ) -> None:
        if cache:
            dataset = tuple(entry for entry in dataset)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> any:
        return self.dataset[index]
