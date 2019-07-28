import os
import numpy as np
# from IPython.core.debugger import Tracer
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D
from keras.layers import BatchNormalization,Embedding,multiply
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.models import Sequential, load_model,Model
from keras.optimizers import RMSprop
import sys
import keras.backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

d_iter = 5


def wloss(y_true, y_pred):
    V = K.mean(K.abs(y_pred - y_true), axis=-1)
    return V


def save_imgs(generator, Number):
    import matplotlib.pyplot as plt

    # 5 tags at the same time
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    label = [32,32,32,32,32, 45,45,45,45,45, 80,80,80,80,80, 26,26,26,26,26, 73,73,73,73,73]
    label = np.array(label).reshape(-1, 1)

    gen_imgs = generator.predict([noise, label])
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs.astype(float)

    # bgr -> rgb
    image_list = []
    for i in range(r * c):
        image_list.append(gen_imgs[i, :, :, ::-1])
    image_list = np.array(image_list)
    gen_imgs = image_list

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(os.getcwd() + "/output_cgan/cgan_{}.png".format(Number))
    plt.close()


class GAN(object):

    def __init__(self, width=64, height=64, channels=3):
        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)
        self.num_classes = 120
        self.latent_dim = 100


        self.LossFunction = wloss
        self.n_critic = 5
        self.clip_value = 0.01
        self.optimizer = RMSprop(lr=0.00005)
        self.D = self.__discriminator()
        self.D.compile(loss=self.LossFunction, optimizer=self.optimizer, metrics=['accuracy'])

        self.G = self.__generator()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.G([noise, label])
        self.D.trainable = False
        valid = self.D([img, label])
        self.G_D = Model([noise, label], valid)
        self.G_D.compile(loss=self.LossFunction,
                              optimizer=self.optimizer)
    def __generator(self):
        """ Declare generator """

        generator = Sequential(name="Generator")
        generator.add(Dense(128 * 16 * 16, input_shape=(100,)))
        generator.add(ReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Reshape((16, 16, 128)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=4))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(UpSampling2D())
        generator.add(Conv2D(64, kernel_size=4))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Conv2D(3, kernel_size=4))
        generator.add(LeakyReLU())
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Flatten())
        generator.add(Dense(64 * 64 * 3, activation='tanh'))
        generator.add(Reshape((64, 64, 3)))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = generator(model_input)

        return Model([noise, label], img)

    def __discriminator(self):
        """ Declare discriminator """

        discriminator = Sequential(name="Discriminator")
        discriminator.add(Dense(64 * 16 * 16, input_dim=np.prod(self.shape)))
        discriminator.add(LeakyReLU())
        discriminator.add(Reshape((16, 16, 64)))

        discriminator.add(Conv2D(32, kernel_size=4, input_shape=(64, 64, 3) ))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(64, kernel_size=4, padding='same' ))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(128, kernel_size=4 ))
        discriminator.add(LeakyReLU())
        discriminator.add(Conv2D(256, kernel_size=4 ))
        discriminator.add(LeakyReLU())

        discriminator.add(Flatten())
        discriminator.add(Dense(1))

        img = Input(shape=self.shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = discriminator(model_input)

        return Model([img, label], validity)



    def train(self, X_train,y_train, epochs=120000, batch=64):
        valid = np.ones((batch, 1))
        fake = -np.ones((batch, 1))
        for cnt in range(epochs):
            for i in range(0, self.n_critic):
                idx = np.random.randint(0, X_train.shape[0], batch)
                imgs, labels = X_train[idx], y_train[idx]
                noise = np.random.normal(0, 1, (batch, 100))
                gen_imgs = self.G.predict([noise, labels])

                d_loss_real = self.D.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.D.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                for l in self.D.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # Condition on labels
            for i in range(0, self.n_critic):
                sampled_labels = np.random.randint(0, 119, batch).reshape(-1, 1)
                noise = np.random.normal(0, 1, (batch, 100))
                g_loss = self.G_D.train_on_batch([noise, sampled_labels], valid)

            save_imgs(self.G, cnt)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (cnt, d_loss[0], 100*d_loss[1], g_loss))

            if (cnt) % 1000 == 0 and cnt != 0:
                self.D.save(os.getcwd() + '/cwgan_model/cgan_D_{}.h5'.format(cnt + 1))
                self.G.save(os.getcwd() + '/cwgan_model/cgan_G_{}.h5'.format(cnt + 1))

 


if __name__ == '__main__': 

    IMGs = np.load('real_images.npy')
    # Rescale -1 to 1
    X_train = (IMGs.astype(np.float32) - 127.5) / 127.5
    y_train = np.load('tags.npy')

    gan = GAN()
    gan.train(X_train,y_train)
