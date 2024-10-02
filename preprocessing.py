import ants
import tensorflow as tf
from PIL import Image
import numpy as np


class ImagePreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None

    def load_image(self):
        """Load the PNG image using PIL and convert to NumPy array."""
        img = Image.open(self.image_path).convert('L')
        self.image = np.array(img)  # Convert the image to a NumPy array
        return self

    def resample_image(self, size=(128, 128)):
        """Resample the image to a new size."""
        if self.image is not None:
            # Resize using PIL for PNG images
            img = Image.fromarray(self.image)
            img = img.resize(size)
            self.image = np.array(img)
        return self

    def apply_smoothing(self, sigma=1.0):
        """Apply Gaussian smoothing using ANTsPy by converting the PNG to an ANTsPy image."""
        if self.image is not None:
            # Convert the NumPy image to ANTsPy image
            ants_image = ants.from_numpy(self.image)
            smoothed_image = ants.smooth_image(ants_image, sigma=sigma)
            self.image = smoothed_image.numpy()  # Convert back to NumPy
        return self

    def normalize_image(self):
        """Normalize the image pixel values to the range [-1, 1]."""
        if self.image is not None:
            # Normalize NumPy array values to [-1, 1]
            self.image = (self.image / 127.5) - 1
        return self

    def to_tensor(self):
        """Convert the preprocessed NumPy image to a TensorFlow tensor."""
        if self.image is not None:
            return tf.convert_to_tensor(self.image, dtype=tf.float32)
        return None

    def augment_tensor(self, tensor):
        """Apply TensorFlow-based augmentations."""
        tensor = tf.image.random_flip_left_right(tensor)
        return tensor

    def preprocess_image_train(self):
        """Pipeline for preprocessing the image with PIL, ANTsPy, and TensorFlow."""
        # Load and preprocess image with PIL and ANTsPy
        self.load_image().resample_image().apply_smoothing().normalize_image()

        # Convert to TensorFlow tensor
        image_tensor = self.to_tensor()

        # Apply TensorFlow augmentation
        image_tensor = self.augment_tensor(image_tensor)

        return image_tensor
