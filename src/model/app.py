from metrics import *
from tqdm import tqdm
from train import *
from tensorflow.keras.utils import CustomObjectScope
import cv2
import numpy as np
import matplotlib.pyplot as plt

x = r"C:\Users\erzou\Desktop\lUNG_CANCER\eval_data\image1\LIDC-IDRI-0001-87_0.png"
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model(r"C:\Users\erzou\Desktop\lUNG_CANCER\files\model.h5")
image = cv2.imread(x, cv2.IMREAD_COLOR)

x = image / 255.0
x = np.expand_dims(x, axis=0)

y_pred = model.predict(x)[0]
y_pred = np.squeeze(y_pred, axis=-1)
y_pred = y_pred > 0.5
y_pred = y_pred.astype(np.int32)
y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)

y_pred = y_pred * 255
# Convert predicted mask to uint8 for visualization
y_pred_vis = y_pred.astype(np.uint8)

# Find contours of the mask
contours, _ = cv2.findContours(y_pred_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw only the contours without filling them
contour_image = np.zeros_like(image, dtype=np.uint8)
cv2.drawContours(contour_image, contours, -1, (0, 0, 255), thickness=2)  # Draw contours with blue color and thickness 2

# Combine the contour image with the original image to show the contours
contour_overlay = cv2.addWeighted(image, 1, contour_image, 0.5, 0)

# Display the result
cv2.imwrite("results/test2.png",contour_overlay)
