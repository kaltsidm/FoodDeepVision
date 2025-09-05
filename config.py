import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Data configuration
DATA_NAME = "food101"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 101

# Model configuration
INPUT_SHAPE = IMG_SIZE + (3,)
BASE_LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2

# Training configuration
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Paths
CHECKPOINT_PATH = "models/checkpoint.ckpt"
SAVED_MODEL_PATH = "models/food_vision_model.h5"
TENSORBOARD_DIR = "logs/"

# Data augmentation
AUGMENTATION = True
ROTATION_RANGE = 0.2
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.2
BRIGHTNESS_RANGE = [0.8, 1.2]