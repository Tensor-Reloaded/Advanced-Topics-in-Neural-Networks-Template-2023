from PIL import Image
import os


class DataReader:
    def __init__(self, folder_path, time_skip):
        self.folder_path = folder_path
        self.image_dictionary = {}
        self.time_skip = time_skip
        self.dataset = None

    def read_and_convert_images(self):
        # We create a dictionary with the key being the path to each folder containing images
        # and the value being a list of numpy arrays (each one representing an image)
        dictionary_image = {}
        for root, dirs, _ in os.walk(self.folder_path):
            for folder in dirs:
                if folder == "images":
                    dictionary_image[os.path.join(root, folder)] = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".tif"):
                    file_path = os.path.join(root, file)
                    dictionary_image[root].append(Image.open(file_path))
        self.image_dictionary = dictionary_image

    def split_by_time_skip(self, images_list):
        # Using the time skip we create tuples with:
        # the first element being an image taken at a specific month (nr)
        # the second element being an image taken at the nr + time_skip month
        # the third element being the time skip between those two images
        tuples_list = []
        for i in range(0, len(images_list) - self.time_skip):
            tuples_list.append((images_list[i], images_list[i + self.time_skip], self.time_skip))
        return tuples_list

    def create_dataset(self):
        dataset_list = []
        if self.time_skip < 1:
            self.time_skip = 1
        for images in self.image_dictionary.values():
            dataset_list += self.split_by_time_skip(images)
        self.dataset = dataset_list
