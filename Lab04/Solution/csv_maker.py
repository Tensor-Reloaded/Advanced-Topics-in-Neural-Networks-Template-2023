import os

path_to_csv = "./data/images.csv"
path_to_images = "../Homework Dataset"

def get_date_diff(year1, month1, year2, month2):
    return (year2 - year1) * 12 + (month2 - month1)

def get_image_paths(path_to_csv):
    with open(path_to_csv, "w") as f:
        for root, dirs, files in os.walk(path_to_images):
            for dir in dirs:
                images_path = os.path.join(root, dir, "images")
                image_list = os.listdir(images_path)
                for i in range(len(image_list)):
                    for j in range(len(image_list)):
                        image1 = image_list[i]
                        image2 = image_list[j]
                        image1_year = int(image1.split("_")[2])
                        image1_month = int(image1.split("_")[3])
                        image2_year = int(image2.split("_")[2])
                        image2_month = int(image2.split("_")[3])
                        diff = get_date_diff(image1_year, image1_month, image2_year, image2_month)
                        image1_path = os.path.join(root, dir, "images", image1)
                        image2_path = os.path.join(root, dir, "images", image2)
                        if diff > 0:
                            f.write(f"{image1_path},{diff},{image2_path}\n")

get_image_paths(path_to_csv)
