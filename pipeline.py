import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.utils import plot_model
import pathlib
import imageio
import glob
import PIL
from data_processing import ImagePreprocessor

import os
root_path = '/path_to_dataset'  # Change this path to that of dataset
os.chdir(root_path)

data_dir_t1 = pathlib.Path("./Tr1/")
data_dir_t2 = pathlib.Path("./Tr2/")
print("T1 MRI images: ", len(list(data_dir_t1.glob('*/*.png'))))
print("T2 MRI images: ", len(list(data_dir_t2.glob('*/*.png'))))

# Initializing constants
BUFFER_SIZE = 1000
BATCH_SIZE = 16
EPOCHS = 125
img_height = 256
img_width = 256

# T1 MRI images Train set
tr1_train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_t1,
    seed=123,
    validation_split=0.075,
    subset='training',
    labels=None,
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE)

# TR1 Test set
tr1_test = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_t1,
    seed=123,
    validation_split=0.075,
    subset='validation',
    image_size=(img_height, img_width),
    batch_size=1)

# TR2 Train set
tr2_train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_t2,
    seed=123,
    validation_split=0.07,
    subset='training',
    labels=None,
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE)

# TR2 Test set
tr2_test = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_t2,
    seed=123,
    validation_split=0.07,
    subset='validation',
    image_size=(img_height, img_width),
    batch_size=1)

AUTOTUNE = tf.data.experimental.AUTOTUNE
tr1_train = tr1_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
tr1_test = tr1_test.cache().prefetch(buffer_size=AUTOTUNE)

tr2_train = tr2_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
tr2_test = tr2_test.cache().prefetch(buffer_size=AUTOTUNE)

preprocessor = ImagePreprocessor('./path_to_image.png')
preprocessed_image = preprocessor.preprocess_image_train(grayscale=True)


tr1_train = tr1_train.map(lambda x: (preprocessor.preprocess_image_train(x)))
tr2_train = tr2_train.map(lambda x: (preprocessor.preprocess_image_train(x)))
tr1_test = tr1_test.map(lambda x, _: (preprocessor.preprocess_image_train(x)))
tr2_test = tr2_test.map(lambda x, _: (preprocessor.preprocess_image_train(x)))

image_batch_tr1 = next(iter(tr1_train))
image_batch_tr2 = next(iter(tr2_train))
tr1_1 = image_batch_tr1[0]
tr2_1 = image_batch_tr2[0]

sample_tr1 = next(iter(tr1_train))
sample_tr2 = next(iter(tr2_train))
