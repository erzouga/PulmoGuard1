
import cv2
import os
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, split=0.2):
    """ Load the images and masks """
    images = sorted(glob(f"{path}/image/*.png"))
    masks = sorted(glob(f"{path}/mask/*.png"))

    """ Split the data """
    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)
    return (train_x, train_y), (valid_x, valid_y)


def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the dir name and image name """
        name = x.split('\\')[1].split('.')[0]
        """ Read the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            X = [x, x1, x2]
            Y = [y, y1, y2]
        else:
            X = [x]
            Y = [y]

        idx = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))
            m = m / 255.0
            m = (m > 0.5) * 255

            if len(X) == 1:
                tmp_image_name = f"{name}.png"
                tmp_mask_name = f"{name}.png"
            else:
                tmp_image_name = f"{name}_{idx}.png"
                tmp_mask_name = f"{name}_{idx}.png"

            image_path = os.path.join(save_path, "image/", tmp_image_name)
            mask_path = os.path.join(save_path, "mask/", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

