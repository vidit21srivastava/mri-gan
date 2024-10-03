import tensorflow as tf

LAMBDA = 10.0

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    return (real_loss + generated_loss) * 0.5


def calc_cycle_loss(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))
