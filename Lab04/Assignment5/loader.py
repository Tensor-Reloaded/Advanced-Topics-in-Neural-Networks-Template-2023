import pandas as pd
from torch.utils.data import DataLoader

from Assignment5.dataset import Dataset


class CustomDataLoaderCSV(DataLoader):
    def __init__(self, dataset: str, transformations=None, **kwargs):
        df = pd.read_csv(dataset)
        df = df.values.tolist()
        df = Dataset(dataset, lambda path: (df, len(df)), transformations=transformations)
        super().__init__(df, **kwargs)


class TrainLoader(CustomDataLoaderCSV):
    def __init__(self, dataset: str = 'train.csv', transformations=None, **kwargs):
        super().__init__(dataset, transformations, **kwargs)


class TestLoader(CustomDataLoaderCSV):
    def __init__(self, dataset: str = 'test.csv', transformations=None, **kwargs):
        super().__init__(dataset, transformations, **kwargs)


class ValidateLoader(CustomDataLoaderCSV):
    def __init__(self, dataset: str = 'validate.csv', transformations=None, **kwargs):
        super().__init__(dataset, transformations, **kwargs)
