import pathlib
import tensorflow as tf
from data_processing import load_datasets
from mri_gan import unet_generator, discriminator
from loss import *
from training import train_step
from plotting import generate_images


class MRIGANPipeline:
    def __init__(self, data_dir_t1, data_dir_t2, img_height, img_width, batch_size, buffer_size, epochs):
        self.data_dir_t1 = pathlib.Path(data_dir_t1)
        self.data_dir_t2 = pathlib.Path(data_dir_t2)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs

        # Initialize models, optimizers, and datasets
        self.generator_g = unet_generator()
        self.generator_f = unet_generator()
        self.discriminator_x = discriminator()
        self.discriminator_y = discriminator()

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)

        self.tr1_train, self.tr2_train, self.tr1_test, self.tr2_test = load_datasets(
            self.data_dir_t1, self.data_dir_t2, self.img_height, self.img_width, self.batch_size, self.buffer_size)

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def train_epoch(self):
        # Loop through the dataset and perform training step
        for image_x, image_y in tf.data.Dataset.zip((self.tr1_train, self.tr2_train)):
            gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = train_step(
                self.generator_g, self.generator_f,
                self.discriminator_x, self.discriminator_y,
                self.generator_g_optimizer, self.generator_f_optimizer,
                self.discriminator_x_optimizer, self.discriminator_y_optimizer,
                image_x, image_y, self.loss_obj
            )
        return gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss

    def generate_and_visualize(self, epoch):
        # Generate and visualize images after each epoch
        sample_tr1 = next(iter(self.tr1_train))
        sample_tr2 = next(iter(self.tr2_train))
        generate_images(self.generator_g, sample_tr1,
                        self.generator_f, sample_tr2, epoch)

    def train(self):
        # Train for the specified number of epochs
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = self.train_epoch()

            # Print the loss for the epoch
            print(
                f"Gen G Loss: {gen_g_loss}, Gen F Loss: {gen_f_loss}, Disc X Loss: {disc_x_loss}, Disc Y Loss: {disc_y_loss}")

            # Generate and visualize the results
            self.generate_and_visualize(epoch)

        print("Training completed!")
