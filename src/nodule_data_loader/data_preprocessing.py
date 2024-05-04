import pylidc as pl
import cv2
import os
import pydicom
from pylidc.utils import consensus
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

pids =[]
for i in range(500):
    if i> 98:
        pids.append(f'LIDC-IDRI-0{i + 1}')
    elif i > 8:
        pids.append(f'LIDC-IDRI-00{i + 1}')

    else:
        pids.append(f'LIDC-IDRI-000{i + 1}')
print(pids)

pids.remove('LIDC-IDRI-0238')

folder = 'test3/train/image/'
create_dir(folder)
scans = pl.query(pl.Scan).all()

# if we have all the data
# for s in scans:
#   scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == s.patient_id).first()
for pid in pids:
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    dicom_images = scan.load_all_dicom_images()
    i = 0

    for dicom_image in dicom_images:
        # Normalize pixel values between 0 and 255
        normalized_image = ((dicom_image.pixel_array - np.min(dicom_image.pixel_array)) /
                            (np.max(dicom_image.pixel_array) - np.min(dicom_image.pixel_array)) * 255.0)

        # Convert pixel values to uint8
        normalized_image = normalized_image.astype(np.uint8)

        # Save the image
        i += 1
        name = f"{folder}{pid}-{i}.png"
        cv2.imwrite(name, normalized_image)
        print(name)

def mask_to_png(pid):
    annotations_mask_512x512=[]
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    nods = scan.cluster_annotations()
    vol = scan.to_volume()
    mask_512x512_vol = np.zeros(vol.shape, dtype=np.uint8)
    for i in nods:
        masks = i[0].boolean_mask()
        bbox = i[0].bbox()
        relevant_vol = vol[bbox]
        num_slices = min(relevant_vol.shape[2], masks.shape[2])
        # Dimensions de l'image 512x512
        mask_shape =(512,512,num_slices)
        # Initialiser le masque binaire de l'image 512x512 avec des zéros
        mask_512x512_num_slices = np.zeros(mask_shape, dtype=np.uint8)
        # Insérer le masque du relevant_vol dans la région de la boîte englobante de l'image 512x512
        for j in range(num_slices):
            mask_512x512_num_slices[bbox[0], bbox[1],j] = i[0].boolean_mask()[:,:,j]

        mask_512x512_vol[:, :,bbox[2]]=np.maximum(mask_512x512_num_slices, mask_512x512_vol[:, :,bbox[2]])
    return (mask_512x512_vol)

folder='test3/train/mask/'
create_dir(folder)
for pid in pids:
    print(pid)
    mask_vol=mask_to_png(pid)
    for i in range (mask_vol.shape[2]):
        # Afficher l'image DICOM
        image=mask_vol[:,:,i]*255.0
        i+=1
        name=f"{folder}{pid}-{i}.png"
        cv2.imwrite(name,image)
        print(name)

images = sorted(glob("C:/Users/erzou/Desktop/lUNG_CANCER/test3/train/image/*.png"))
print(len(images))
masks = sorted(glob("C:/Users/erzou/Desktop/lUNG_CANCER/test3/train/mask/*.png"))
print(len(masks))
image_n_path = []
mask_n_path = []
image_nod_path = []
mask_nod_path = []

save_path = "C:/Users/erzou/Desktop/lUNG_CANCER/test3/train"

for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
    """ Extracting the dir name and image name """
    name = x.split('\\')[-1].split('.')[0]
    """ Read the image and mask """

    y = cv2.imread(y, cv2.IMREAD_COLOR)
    tmp_image_name = f"{name}.png"
    tmp_mask_name = f"{name}.png"
    if np.max(y) == 255:
        image_nod_path.append(os.path.join(save_path, "image/", tmp_image_name))
        mask_nod_path.append(os.path.join(save_path, "mask/", tmp_mask_name))
    else:
        image_n_path.append(os.path.join(save_path, "image/", tmp_image_name))
        mask_n_path.append(os.path.join(save_path, "mask/", tmp_mask_name))

print(mask_nod_path)

path = "C:/Users/erzou/Desktop/lUNG_CANCER/test3/train2"
path = "C:/Users/erzou/Desktop/lUNG_CANCER/test3/train2"

create_dir("test3/train2/image_nod")
create_dir("test3/train2/mask_nod")

for idx, (x, y) in tqdm(enumerate(zip(image_nod_path, mask_nod_path))):
    """ Extracting the dir name and image name """
    name = x.split('/')[-1].split('.')[0]
    print(x)
    """ Read the image and mask """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    y = cv2.imread(y, cv2.IMREAD_COLOR)

    tmp_image_name = f"{name}.png"
    tmp_mask_name = f"{name}.png"
    im_path = os.path.join(path, "image_nod/", tmp_image_name)
    m_path = os.path.join(path, "mask_nod/", tmp_mask_name)
    print(im_path)

    cv2.imwrite(im_path, x)
    cv2.imwrite(m_path, y)