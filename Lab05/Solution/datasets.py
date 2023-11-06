from torch.utils.data import Dataset

class CachedDataset(Dataset):
    def __init__(self, dataset):
        dataset = tuple([x for x in dataset])
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]
    
class WithTransform(Dataset):
    def __init__(self, dataset, transform):
        dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        input, target = self.dataset[i]
        input = self.transform(input)
        return input, target
