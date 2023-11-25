class Dataset:
    def __init__(self, dataset_path, build_dataset, transformations=None,
                 training=True, transformations_test=None):
        self.transformations_train = transformations if transformations is not None else []
        self.transformations_test = transformations_test if transformations_test is not None else []
        self.photo_pairs, self.dataset_size = build_dataset(dataset_path)
        self.training = training
        # photo pairs can be any pair input-output, not necessarily consisting in photos

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        photo_pair = self.photo_pairs[idx]  # Don't transform the original features
        if self.training:
            for transform in self.transformations_train:
                photo_pair = transform(photo_pair)
        else:
            for transform in self.transformations_test:
                photo_pair = transform(photo_pair)
        return photo_pair
