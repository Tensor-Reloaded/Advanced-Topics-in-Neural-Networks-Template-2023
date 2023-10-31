import pandas as pd
from torch.utils.data import DataLoader

from Assignment5.dataset import Dataset


class CustomDataLoaderCSV(DataLoader):
    def __init__(self, dataset: str, transformations=None, training=True,
                 transformations_test=None, **kwargs):
        df = pd.read_csv(dataset)
        df = df.values.tolist()
        df = Dataset(dataset, lambda path: (df, len(df)),
                     transformations=transformations,
                     transformations_test=transformations_test,
                     training=training)
        super().__init__(df, **kwargs)


class TrainLoader(CustomDataLoaderCSV):
    def __init__(self, dataset: str = 'train.csv',
                 transformations=None, transformations_test=None, **kwargs):
        super().__init__(dataset, transformations,
                         transformations_test=transformations_test, **kwargs)


class TestLoader(CustomDataLoaderCSV):
    def __init__(self, dataset: str = 'test.csv',
                 transformations=None, transformations_test=None, **kwargs):
        super().__init__(dataset, transformations, training=False,
                         transformations_test=transformations_test, **kwargs)


class ValidateLoader(CustomDataLoaderCSV):
    def __init__(self, dataset: str = 'validate.csv',
                 transformations=None, transformations_test=None, **kwargs):
        super().__init__(dataset, transformations, training=False,
                         transformations_test=transformations_test, **kwargs)
