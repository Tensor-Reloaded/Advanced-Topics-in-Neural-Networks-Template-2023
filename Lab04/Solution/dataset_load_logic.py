import os
from datetime import date


class Picture:
    def __init__(self, picture_path: str, picture_date: date):
        self.picture_path = picture_path
        self.picture_date = picture_date

    @staticmethod
    def extract_date(image_name: str) -> date:
        assert image_name.startswith("global_monthly_")
        image_name = image_name[len("global_monthly_"):]
        image_name_parts = image_name.split("_")
        assert len(image_name_parts) > 2
        return date(int(image_name_parts[0]), int(image_name_parts[1]), 1)

    @staticmethod
    def from_images_path(images_path: str) -> list["Picture"]:
        pictures = []

        images_path = os.path.join(images_path, "images")
        assert os.path.exists(images_path)

        for image_name in os.listdir(images_path):
            if not os.path.isfile(os.path.join(images_path, image_name)):
                continue

            pictures.append(Picture(os.path.join(images_path, image_name), Picture.extract_date(image_name)))
        return pictures


class SatelliteLocation:
    def __init__(self, location_name: str, pictures: list[Picture]):
        self.location_name = location_name
        self.pictures = list(sorted(pictures, key=lambda obj: obj.picture_date))

    @staticmethod
    def from_dir_path(dir_path: str) -> list["SatelliteLocation"]:
        satellite_locations = []
        for dir_name in os.listdir(dir_path):
            if not os.path.isdir(os.path.join(dir_path, dir_name)):
                continue

            pictures = Picture.from_images_path(os.path.join(dir_path, dir_name))
            satellite_locations.append(SatelliteLocation(dir_name, pictures))
        return satellite_locations


def load_data(dir_path: str) -> list[tuple[str, str, int]]:
    satellite_locations = SatelliteLocation.from_dir_path(dir_path)

    img_data = []
    for location in satellite_locations:
        for idx, lh_picture in enumerate(location.pictures[:-1]):
            for rh_picture in location.pictures[idx+1:]:
                img_data.append(
                    (lh_picture.picture_path,
                     rh_picture.picture_path,
                     int(round((rh_picture.picture_date - lh_picture.picture_date).days / 30)))
                )

    return img_data

