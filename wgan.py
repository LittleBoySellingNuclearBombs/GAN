import os
import numpy as np 
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
import sys
import keras.backend as K
import tensorflow as tf 
#setting GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
import keras.backend.tensorflow_backend as KTF
import keras.losses 
KTF.set_session(session )

d_iter=5

def wloss(y_true, y_pred):
    V = K.mean(K.abs(y_pred - y_true), axis=-1)
    return V
def save_imgs(generator,Number):
    import matplotlib.pyplot as plt

    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = (generator.predict(noise))
    #range 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    gen_imgs.astype(float)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(os.getcwd()+"/output_cgan_demo/output_wgan_{}.png".format(Number))
    plt.close()

class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=64, height=64, channels=3):
        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)
        self.LossFunction = wloss
        self.n_critic = 5
        # setting the weight between -clip_value and clip_value
        self.clip_value = 0.01
        #change the optimizer
        self.optimizer = RMSprop(lr=0.00005)
        self.G = self.__generator()
        #self.G.load_weights(os.getcwd() + '/wgan_model/wgan_G.h5')
        self.G.compile(loss=self.LossFunction, optimizer=self.optimizer)

        self.D = self.__discriminator()
        #self.D.load_weights(os.getcwd() + '/wgan_model/wgan_D.h5')
        self.D.compile(loss=self.LossFunction, optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()
        self.stacked_generator_discriminator.compile(loss=self.LossFunction, optimizer=self.optimizer)

    def __generator(self): 
        generator=Sequential(name="Generator")
        generator.add(Dense(128*16*16, input_shape=(100,), activation='relu'))


        generator.add(Reshape((16,16,128)))
        generator.add(UpSampling2D())
        generator.add(Conv2D(128, kernel_size=4, activation='relu'))


        generator.add(UpSampling2D())
        generator.add(Conv2D(64, kernel_size=4, activation='relu'))


        generator.add(Conv2D(3, kernel_size=4, activation='relu'))

        generator.add(Flatten())
        generator.add(Dense(64*64*3, activation='tanh'))
        generator.add(Reshape((64,64,3)))


        return  generator

    def __discriminator(self): 
        discriminator=Sequential(name="Discriminator")
        discriminator.add(Conv2D(32, kernel_size=4, input_shape=(64,64,3), activation='relu'))
        
        discriminator.add(Conv2D(64, kernel_size=4, padding='same', activation='relu'))
        
        discriminator.add(Conv2D(128, kernel_size=4, activation='relu'))
        
        discriminator.add(Conv2D(256, kernel_size=4, activation='relu'))
        
        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        # remove the activity function sigmoid
        return discriminator


    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        G_D_model = Sequential()
        G_D_model.add(self.G)
        G_D_model.add(self.D)

        return G_D_model

    

    def train(self, X_train, epochs=120000, batch = 64):

        for cnt in range(epochs):
            for i in range(0,self.n_critic):
                self.D.trainable = True
                random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
                legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)

                gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
                syntetic_images = self.G.predict(gen_noise)

                x_combined_batch = np.concatenate((legit_images, syntetic_images))
                y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

                d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
                #setting the weight between -clip_value and clip_value
                for l in self.D.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
 
            self.D.trainable = False
            noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
            y_mislabled = np.ones((np.int64(batch/2), 1))
            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            save_imgs(self.G, cnt)
            print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

            if (cnt) % 1000 == 0 and  cnt != 0:
                self.D.save(os.getcwd() + '/wgan_model/wgan_D_{}.h5'.format(cnt + 1))
                self.G.save(os.getcwd() + '/wgan_model/wgan_G_{}.h5'.format(cnt + 1))

 
if __name__ == '__main__': 

    IMGs=np.load('real_images.npy')
    # Rescale -1 to 1
    X_train = (IMGs.astype(np.float32) - 127.5) / 127.5

    gan = GAN()
    gan.train(X_train)
