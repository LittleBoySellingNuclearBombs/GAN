import os
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# seeting the gup 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# it can use 80% on the "0"GPU for current code
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session )
# give the noise & create the img 
def save_imgs(generator,Number):
    import matplotlib.pyplot as plt
    r, c = 5, 5
    #creat the noise 
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs =  (generator.predict(noise))
    #reshape the gen_imgs to 0~1 , it already reshape -1 ~ 1 by (IMGs.astype(np.float32) - 127.5) / 127.5
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs.astype(float)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,:])
            axs[i,j].axis('off')
            cnt += 1
    #save path, so u need to create the folder "output"to save the imgs
    fig.savefig(os.getcwd()+"/output/gan_{}.png".format(Number))
    plt.close()
class GAN(object):
     
    def __init__(self, width=64, height=64, channels=3):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        self.G = self.__generator()
        #if u need to reload a model to continue training,the path need to follow your path 
        #self.G.load_weights(os.getcwd() + '/model/gan_G.h5') 
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        #self.D.load_weights(os.getcwd() + '/model/gan_D.h5') 
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()
        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    # create the generator
    def __generator(self):
        generator=Sequential()
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
    # create the discriminator
    def __discriminator(self):
        discriminator=Sequential()
        discriminator.add(Conv2D(32, kernel_size=4, input_shape=(64,64,3), activation='relu'))
        discriminator.add(Conv2D(64, kernel_size=4, padding='same', activation='relu'))
        discriminator.add(Conv2D(128, kernel_size=4, activation='relu'))
        discriminator.add(Conv2D(256, kernel_size=4, activation='relu'))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))

        return discriminator
 
    def __stacked_generator_discriminator(self):
        self.D.trainable = False
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        return model

    def train(self, X_train, epochs=35000, batch = 150):

        for cnt in range(epochs):
            #train discriminator
            for i in range(0,5):
                self.D.trainable = True
                #prevent the random_index beyond the leng of X_train,the data consist of real imgs with 1/2 batch size  & 1/2 fake imgs
                # with 1/2 batch size
                random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
                #Random the index 
                np.random.shuffle(X_train)
                #Get the real images between random_index to random_index + (batch size)/2
                legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)

                gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
                
                syntetic_images = self.G.predict(gen_noise)
                
                x_combined_batch = np.concatenate((legit_images, syntetic_images))
                #1 stand for real imgs , 0 stand for fake 
                y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))
                # training D to identify real or fake imgs
                d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

            # just training G,and make it can cheat D that imgs from real 
            self.D.trainable = False
            noise = np.random.normal(0, 1, (int(batch/2), 100))
            y_mislabled = np.ones((int(batch/2), 1))
            
            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)
            # save imgs
            save_imgs(self.G,cnt)
            
            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
            
            if (cnt) % 2500 == 0 :
                #model path
                self.D.save(os.getcwd() + '/model/gan_D_{}.h5'.format(cnt + 1))
                self.G.save(os.getcwd() + '/model/gan_G_{}.h5'.format(cnt+1 ))

 
if __name__ == '__main__': 
    #u need to creat the folder "output" to store the imgs from G,and folder"model"to store the model
    IMGs=np.load('real_images.npy')#read the real_images.py to creat this file 
    # reshape to -1 - 1
    X_train = (IMGs.astype(np.float32) - 127.5) / 127.5
    gan = GAN()
    gan.train(X_train)
