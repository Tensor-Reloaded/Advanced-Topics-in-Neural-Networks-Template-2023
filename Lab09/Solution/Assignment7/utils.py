import csv
import itertools
import shutil
import os
from typing import Tuple

import torch
import torchvision.utils
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import v2


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_abs_time_difference(date1: Tuple[int, int], date2: Tuple[int, int]) -> int:
    if date1[0] < date2[0]:
        return (12 - date1[1]) + (12 * (date2[0] - date1[0] - 1)) + date2[1]
    elif date1[0] == date2[0] and date1[1] <= date2[1]:
        return date2[1] - date1[1]
    elif date1[0] > date2[0]:
        return (12 - date2[1]) + (12 * (date1[0] - date2[0] - 1)) + date1[1]
    elif date1[0] == date2[0] and date1[1] > date2[1]:  # else
        return date1[1] - date2[1]


def split_dataset(dataset, split_root):  # build new folder structure
    if os.path.exists(split_root):
        return
    os.mkdir(split_root)
    os.mkdir(os.path.join(split_root, 'train'))
    os.mkdir(os.path.join(split_root, 'test'))
    os.mkdir(os.path.join(split_root, 'validate'))
    x_train, x_test = train_test_split(dataset, train_size=0.7, random_state=42)
    x_test, x_validation = train_test_split(x_test, train_size=0.5, random_state=42)

    training_path = os.path.join(split_root, 'train')
    for instance in x_train:
        generate_files(instance, training_path)
    testing_path = os.path.join(split_root, 'test')
    for instance in x_test:
        generate_files(instance, testing_path)
    validation_path = os.path.join(split_root, 'validate')
    for instance in x_validation:
        generate_files(instance, validation_path)


def generate_files(instance, path):
    id_pair = instance[0].split('global_monthly_')[1][:7] + '-' + instance[1].split('global_monthly_')[1][:7]
    split_instance = instance[0][instance[0].find('L15-'):].split('\\')
    new_path = os.path.join(path, split_instance[0], id_pair, split_instance[2])
    try:
        shutil.copyfile(instance[0], new_path)
    except IOError:
        os.makedirs(os.path.dirname(new_path))
        shutil.copyfile(instance[0], new_path)
    split_instance = instance[1][instance[1].find('L15-'):].split('\\')
    new_path = os.path.join(path, split_instance[0], id_pair, split_instance[2])
    shutil.copyfile(instance[1], new_path)


def split_dataset_csv(dataset):
    x_train, x_test = train_test_split(dataset, train_size=0.7, random_state=42)
    x_test, x_validation = train_test_split(x_test, train_size=0.5, random_state=42)

    # csv for training data
    with open('train.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image_in', 'image_out', 'time_elapsed'])
        for row in x_train:
            csv_out.writerow(row)
    # csv for testing data
    with open('test.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image_in', 'image_out', 'time_elapsed'])
        for row in x_test:
            csv_out.writerow(row)
    # csv for validation data
    with open('validation.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image_in', 'image_out', 'time_elapsed'])
        for row in x_validation:
            csv_out.writerow(row)


def build_dataset_initial(dataset_path):
    dataset_size = 0
    photo_pairs = []
    for (root, directories, files) in os.walk(dataset_path):
        photos = []
        for photo in files:
            photo_path = os.path.join(root, photo)
            year_month = os.path.splitext(photo)[0].split('global_monthly_')[1][:7]
            photos.append((photo_path, year_month))
        # build tuples
        pairs = list(itertools.combinations(photos, 2))
        for pair in pairs:
            photo_1 = pair[0][0]
            photo_2 = pair[1][0]
            months = compute_abs_time_difference((int(pair[0][1][:4]), int(pair[0][1][5:])),
                                                 (int(pair[1][1][:4]), int(pair[1][1][5:])))
            photo_pairs.append([photo_1, photo_2, months])
        dataset_size += len(pairs)
    return photo_pairs, dataset_size


def count_correct_predictions(output, labels):
    return Tensor([labels.argmax(dim=1)[input] in output.topk(5, 1, True, True).indices[input]
                   for input in range(labels.shape[0])]).sum().item()


def apply_cosine_similarity(output, labels):
    return (torch.nn.CosineSimilarity()(output.reshape(-1, 28 * 28),
                                        labels.reshape(-1, 28 * 28)) >= 0.95).sum().item()


def turn_to_zero(input: Tensor):
    input[:] = 0


def build_transformed_dataset(dataset, train):
    new_dataset = []
    for index, elem in enumerate(dataset):
        transformed = v2.Compose([
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
            torch.flatten,
            # transforms.ToTensor()
        ])(dataset[index][0])
        new_dataset.append((transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                            (dataset[index][0]),
                            transformed))
        if train:
            torchvision.utils.save_image(transformed.reshape(28, 28), f'Labels\\output_{index}.png')
    return new_dataset, len(new_dataset)
