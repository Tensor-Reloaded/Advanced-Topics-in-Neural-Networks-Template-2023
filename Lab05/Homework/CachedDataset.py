from torch.utils.data import Dataset

class CachedDataset(Dataset):
    def __init__(self, dataset, transforms=None, cache=True):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        if self.transforms is not None:
            return self.transforms(self.dataset[i][0]), self.dataset[i][1]
        else:
            return self.dataset[i]