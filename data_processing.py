import ants
import tensorflow as tf
import pathlib
import numpy as np
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_png_image(image_path):

    image = Image.open(image_path).convert('L')
    image_np = np.array(image).astype(np.float32)
    return image_np


def preprocess_with_antspy(image_np):

    image = ants.from_numpy(image_np)

    corrected_image = ants.n4_bias_field_correction(image)

    normalized_image = ants.iMath(corrected_image, "Normalize")

    return normalized_image


def resize_image(image, target_size):

    resampled_image = ants.resample_image(
        image, target_size, use_voxels=True, interp_type=1)
    return resampled_image


def load_and_preprocess_ants_images(image_path, img_height, img_width):

    image_np = load_png_image(image_path)

    preprocessed_image = preprocess_with_antspy(image_np)

    resized_image = resize_image(preprocessed_image, [img_height, img_width])

    return resized_image.numpy()


def load_datasets(data_dir_t1, data_dir_t2, img_height, img_width, batch_size, buffer_size):

    t1_image_paths = list(pathlib.Path(data_dir_t1).glob('*/*.png'))
    t2_image_paths = list(pathlib.Path(data_dir_t2).glob('*/*.png'))

    def process_image(image_path):
        image_path_str = image_path.numpy().decode('utf-8')
        processed_image = load_and_preprocess_ants_images(
            image_path_str, img_height, img_width)
        return processed_image

    def tf_process_image(image_path):
        return tf.py_function(func=process_image, inp=[image_path], Tout=tf.float32)

    tr1_dataset = tf.data.Dataset.from_tensor_slices(
        [str(p) for p in t1_image_paths])
    tr1_train = tr1_dataset.map(tf_process_image, num_parallel_calls=AUTOTUNE).batch(
        batch_size).shuffle(buffer_size).prefetch(AUTOTUNE)

    tr2_dataset = tf.data.Dataset.from_tensor_slices(
        [str(p) for p in t2_image_paths])
    tr2_train = tr2_dataset.map(tf_process_image, num_parallel_calls=AUTOTUNE).batch(
        batch_size).shuffle(buffer_size).prefetch(AUTOTUNE)

    return tr1_train, tr2_train
