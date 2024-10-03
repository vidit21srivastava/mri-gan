import matplotlib.pyplot as plt


def generate_images(generator_g, test1, generator_f, test2, epoch):
    prediction1 = generator_g(test1)
    prediction2 = generator_f(test2)
    plt.figure(figsize=(8, 8))
    display_list = [test1[0], prediction1[0], test2[0], prediction2[0]]
    title = ['Input T1 Image', 'Translated to T2',
             'Input T2 Image', 'Translated to T1']

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i].numpy()[:, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
