import tensorflow as tf
import pathlib


def normalize(image):
    image = (image / 127.5) - 1
    return image


def preprocess_image_train(image):
    image = tf.image.random_flip_left_right(image)
    image = normalize(image)
    return image


def load_datasets(data_dir_t1, data_dir_t2, img_height, img_width, batch_size, buffer_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Load datasets
    tr1_train = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_t1, seed=123, validation_split=0.075, subset='training', labels=None,
        image_size=(img_height, img_width), batch_size=batch_size)

    tr1_test = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_t1, seed=123, validation_split=0.075, subset='validation',
        image_size=(img_height, img_width), batch_size=1)

    tr2_train = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_t2, seed=123, validation_split=0.07, subset='training', labels=None,
        image_size=(img_height, img_width), batch_size=batch_size)

    tr2_test = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_t2, seed=123, validation_split=0.07, subset='validation',
        image_size=(img_height, img_width), batch_size=1)

    # Apply preprocessing
    tr1_train = tr1_train.map(preprocess_image_train).cache().shuffle(
        buffer_size).prefetch(buffer_size=AUTOTUNE)
    tr2_train = tr2_train.map(preprocess_image_train).cache().shuffle(
        buffer_size).prefetch(buffer_size=AUTOTUNE)
    tr1_test = tr1_test.map(preprocess_image_train).cache().prefetch(
        buffer_size=AUTOTUNE)
    tr2_test = tr2_test.map(preprocess_image_train).cache().prefetch(
        buffer_size=AUTOTUNE)

    return tr1_train, tr2_train, tr1_test, tr2_test
