import os
import re
import datetime
import typing as t
import PIL.Image as PIL_Image
import torch
import torchvision
import torch.utils.data as torch_data
import exceptions

ImageSet = t.Tuple[torch.Tensor, torch.Tensor, int]


class Dataset(torch_data.Dataset):
    __root: str
    __device: str
    __transformations: t.List
    __image_sets: t.List[ImageSet]

    def __init__(
        self,
        root: str,
        transformations: t.List[t.Callable[[torch.Tensor], torch.Tensor]] = [],
        device: str = "cpu",
    ) -> None:
        self.__root = root
        self.__device = device
        self.__transformations = transformations

        self.__image_sets = self.__load(self.__root)
        self.__image_sets = self.__load_on_device(self.__image_sets, self.__device)

    def __load(self, path: str) -> t.List[ImageSet]:
        results: t.List[ImageSet] = []
        image_folder_paths: t.List[t.List[str]] = []
        timestamp_regexp = re.compile("global_monthly_(\d{4}_\d{2})_")

        for folder in os.listdir(path):
            image_folder = []

            for file in os.listdir(f"{path}/{folder}/images"):
                image_folder.append(f"{path}/{folder}/images/{file}")

            image_folder_paths.append(image_folder)

        for folder in image_folder_paths:
            for index in range(0, len(folder) - 1):
                image_file_1 = folder[index]
                image_file_2 = folder[index + 1]
                image_file_1_filename = os.path.basename(image_file_1)
                image_file_2_filename = os.path.basename(image_file_2)
                image_file_1_timestamp_str = timestamp_regexp.search(
                    image_file_1_filename
                )
                image_file_2_timestamp_str = timestamp_regexp.search(
                    image_file_2_filename
                )

                if image_file_1_timestamp_str is None:
                    raise exceptions.DatasetException(
                        f"Image {image_file_1} cannot be parsed for timestamp"
                    )

                if image_file_2_timestamp_str is None:
                    raise exceptions.DatasetException(
                        f"Image {image_file_2} cannot be parsed for timestamp"
                    )

                image_file_1_timestamp = datetime.datetime.strptime(
                    image_file_1_timestamp_str.group(1), "%Y_%m"
                )
                image_file_2_timestamp = datetime.datetime.strptime(
                    image_file_2_timestamp_str.group(1), "%Y_%m"
                )
                image_1 = torchvision.transforms.functional.pil_to_tensor(
                    pic=PIL_Image.open(image_file_1)
                )
                image_2 = torchvision.transforms.functional.pil_to_tensor(
                    pic=PIL_Image.open(image_file_2)
                )
                time_skip = abs(
                    (image_file_1_timestamp.year - image_file_2_timestamp.year) * 12
                    + (image_file_1_timestamp.month - image_file_2_timestamp.month)
                )

                results.append((image_1, image_2, time_skip))

        return results

    def __load_on_device(
        self, image_sets: t.List[ImageSet], device: str
    ) -> t.List[ImageSet]:
        for index in range(0, len(image_sets)):
            image_sets[index] = (
                image_sets[index][0].to(device=device, non_blocking=device == "cuda"),
                image_sets[index][1].to(device=device, non_blocking=device == "cuda"),
                image_sets[index][2],
            )

        return image_sets

    def __len__(self) -> int:
        return len(self.__image_sets)

    def __getitem__(self, index) -> ImageSet:
        image_set = self.__image_sets[index]
        image_1 = image_set[0]
        image_2 = image_set[1]
        time_skip = image_set[2]

        for transform in self.__transformations:
            image_1 = transform(image_1)
            image_2 = transform(image_2)

        return (image_1, image_2, time_skip)

    def get_image_size(self):
        image_sample = self.__image_sets[0][0] 

        for transform in self.__transformations:
            image_sample = transform(image_sample)

        return image_sample.shape[0]