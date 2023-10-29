import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from typing import Optional, Callable
from torch.utils.data import Subset


def same_transform(transform_fn: Callable) -> Callable:
    def inner_same_transform(lh_img: Tensor, rh_img: Tensor) -> tuple[Tensor, Tensor]:
        state = torch.get_rng_state()
        lh_img = transform_fn(lh_img)
        torch.set_rng_state(state)
        rh_img = transform_fn(rh_img)
        return lh_img, rh_img
    return inner_same_transform


COMMON_TRANSFORMS = {
    "random_rotation": lambda arg: same_transform(RandomRotation(degrees=arg))
}


class ImageComparisonDatasetWrapper(Dataset):
    wrapped: bool = False
    load_data_fn: Callable

    @staticmethod
    def set_wrap(this_cls: type["ImageComparisonDatasetWrapper"], load_data_fn: Callable) -> None:
        this_cls.wrapped = True
        this_cls.load_data_fn = lambda *args: load_data_fn(args[-1])

    def __init__(self, path_to_satellite_images: str,
                 transforms_to_apply: Optional[list] = None,
                 **common_transforms):

        assert self.wrapped
        self.images_data_paths = self.load_data_fn(path_to_satellite_images)

        self.transforms = []
        for transform_name, transform_args in common_transforms.items():
            self.transforms.append(COMMON_TRANSFORMS[transform_name](transform_args))

        if transforms_to_apply:
            self.transforms += transforms_to_apply

        self.img_cache = self.__make_cache(self.images_data_paths)

    @staticmethod
    def __make_cache(images_data_paths: list[str, str, int]) -> dict[str, Tensor]:
        result = dict[str, Tensor]()

        for lh_img, rh_img, days_between in images_data_paths:
            if lh_img not in result:
                result[lh_img] = ImageComparisonDatasetWrapper.load_image(lh_img)
            if rh_img not in result:
                result[rh_img] = ImageComparisonDatasetWrapper.load_image(rh_img)

        return result

    def __len__(self) -> int:
        return len(self.images_data_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        lh_img_path, rh_img_path, days_between = self.images_data_paths[idx]
        lh_img, rh_img = self.load_image(lh_img_path), self.load_image(rh_img_path)

        for transform in self.transforms:
            lh_img, rh_img = transform(lh_img, rh_img)

        return lh_img, rh_img, days_between

    @staticmethod
    def load_image(img_path: str) -> Tensor:
        return ToTensor()(Image.open(img_path))

    def split(self, *percentages) -> list[Subset]:
        assert abs(sum(percentages) - 1) < 0.001
        data_amounts = [int(perc * len(self)) for perc in percentages[:-1]]
        data_amounts.append(len(self) - sum(data_amounts))
        return list(torch.utils.data.random_split(self, data_amounts))


def materialize_wrapper(load_data_fn: Callable) -> type["ImageComparisonDatasetWrapper"]:
    class NewWrappedClass(ImageComparisonDatasetWrapper):
        pass

    NewWrappedClass.set_wrap(NewWrappedClass, load_data_fn)
    return NewWrappedClass
