from metrics import *
from src.nodule_data_loader.nodule_data_loader import augment_data

from train import *
from u_net import *

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)



dataset_path = 'C:/Users/erzou/Desktop/lUNG_CANCER/test3/train2'
(train_x, train_y), (valid_x, valid_y) = load_data(dataset_path, split=0.2)

print("Train: ", len(train_x))
print("Valid: ", len(valid_x))

create_dir("new_data3/train/image/")
create_dir("new_data3/train/mask/")
create_dir("new_data3/valid/image/")
create_dir("new_data3/valid/mask/")

augment_data(train_x, train_y, "new_data3/train/", augment=True)
augment_data(valid_x, valid_y, "new_data3/valid/", augment=False)

""" Seeding """
H=512
w=512
np.random.seed(42)
tf.random.set_seed(42)

""" Directory for storing files """
create_dir("files")

""" Hyperparameters """
batch_size = 8
lr = 1e-3
num_epochs = 100
model_path = os.path.join("files", "model.h5")
csv_path = os.path.join("files", "data.csv")
""" dataset """
path = 'C:/Users/erzou/Desktop/lUNG_CANCER'
dataset_path = os.path.join("new_data3")
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
print(dataset_path)
#"C:\Users\erzou\Desktop\lUNG_CANCER\new_data1\train"

train_x, train_y = load_data(train_path)
train_x, train_y = shuffling(train_x, train_y)
valid_x, valid_y = load_data(valid_path)

print(f"Train: {len(train_x)} - {len(train_y)}")
print(f"Valid: {len(valid_x)} - {len(valid_y)}")
train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

model = build_unet((H, W, 3))
metrics = [dice_coef, iou, Recall(), Precision()]
model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
]

model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        shuffle=False
)