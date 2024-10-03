from data_processing import load_datasets
from mri_gan import unet_generator, discriminator
from loss import *
from training import train_step
from plotting import generate_images
import tensorflow as tf

data_dir_t1 = pathlib.Path("./Tr1/")
data_dir_t2 = pathlib.Path("./Tr2/")
img_height, img_width = 256, 256
batch_size, buffer_size = 16, 1000
epochs = 125

tr1_train, tr2_train, tr1_test, tr2_test = load_datasets(
    data_dir_t1, data_dir_t2, img_height, img_width, batch_size, buffer_size)

generator_g = unet_generator()
generator_f = unet_generator()
discriminator_x = discriminator()
discriminator_y = discriminator()

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

for epoch in range(epochs):
    for image_x, image_y in tf.data.Dataset.zip((tr1_train, tr2_train)):
        gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = train_step(
            generator_g, generator_f, discriminator_x, discriminator_y,
            generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer,
            image_x, image_y, loss_obj
        )

    generate_images(generator_g, next(iter(tr1_train)),
                    generator_f, next(iter(tr2_train)), epoch)
