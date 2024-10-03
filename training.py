import tensorflow as tf


def train_step(generator_g, generator_f, discriminator_x, discriminator_y,
               generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer,
               discriminator_y_optimizer, real_x, real_y, loss):

    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # Losses
        gen_g_loss = loss.generator_loss(disc_fake_y)
        gen_f_loss = loss.generator_loss(disc_fake_x)

        total_cycle_loss = loss.calc_cycle_loss(
            real_x, cycled_x) + loss.calc_cycle_loss(real_y, cycled_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + \
            loss.identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + \
            loss.identity_loss(real_x, same_x)

        disc_x_loss = loss.discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = loss.discriminator_loss(disc_real_y, disc_fake_y)

    generator_g_gradients = tape.gradient(
        total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(
        total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(
        disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(
        disc_y_loss, discriminator_y.trainable_variables)

    generator_g_optimizer.apply_gradients(
        zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(
        zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(
        zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(
        zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss
