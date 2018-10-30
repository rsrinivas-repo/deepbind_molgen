import numpy as np
import random

import pandas as pd
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

#import matplotlib.pyplot as plt
#plt.switch_backend('agg')   # allows code to run without a system DISPLAY


class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=28, height=28, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (292)

        #self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8, clipnorm=.5)
        self.optimizer = Adam()
        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)


    def __generator(self):
        """ Declare generator """
        model = Sequential()
        model.add(Dense(356, input_shape=(292,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(356))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(292, activation='tanh'))
        #model.add(Reshape((self.width, self.height, self.channels)))

        return model

    def __discriminator(self):
        """ Declare discriminator """

        model = Sequential()
        model.add(Dense(200, input_shape=(292,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model

    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X_train, epochs=20000, batch = 32, save_interval = 100):

        for cnt in range(epochs):

            ## train discriminator
            random_indexes = random.sample(range(0,x_train.shape[0]),batch)
            #legit_images = X_train[random_index : random_index + batch/2].reshape(batch/2, self.width, self.height, self.channels)

            #gen_noise = np.random.normal(0, 1, (batch/2, 100))
            #syntetic_images = self.G.predict(gen_noise)

            #x_combined_batch = np.concatenate((legit_images, syntetic_images))
            #y_combined_batch = np.concatenate((np.ones((batch/2, 1)), np.zeros((batch/2, 1))))
            x_batch = x_train[random_indexes,:292]
            y_batch = x_train[random_indexes,292]


            d_loss = self.D.train_on_batch(x_batch, y_batch)
            
            print(self.D.test_on_batch(x_batch,y_batch))
            # train generator

            noise = np.random.normal(-1, 1, (batch, 292))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

            
if __name__ == '__main__':

    print("Inside code to train GAN Model")

    validMols = np.load("../data/gan_data_valid.npy")
    #fileName = "../data/validEncodedSMILES.csv"
    #dfValid = pd.read_csv(fileName,delim_whitespace=True,header=None)
    #validMols = dfValid.as_matrix()
    
    
    ones = np.ones((validMols.shape[0],1))
    validMols = np.append(validMols,ones,axis=1)

    print("Shape of VALID array %s "%str(validMols.shape))

    invalidMols = np.load("../data/gan_data_invalid.npy")
    #fileName = "../data/invalidEncodedSMILES.csv"
    #dfInValid = pd.read_csv(fileName,delim_whitespace=True,header=None)
    #invalidMols = dfInValid.sample(n=40000).as_matrix()

    
    zeros = np.zeros((invalidMols.shape[0],1))
    invalidMols = np.append(invalidMols,zeros,axis=1)

    print("Shape of INVALID array %s "%str(invalidMols.shape))
    
    #invalidMols = np.array(random.sample(invalidMols,40000))
    invalidMols = invalidMols[:40000,:]
    print("Shape of INVALID array after sampling %s "%str(invalidMols.shape))
    
    x_train = np.append(validMols,invalidMols,axis=0)
    print ("Shape of FULL training array %s"%str(x_train.shape))
    #np.random.shuffle(x_train)


    gan = GAN()
    gan.train(x_train,epochs=10)
 #gan.train(x_train, epochs=4, batch =10, save_interval = 100)
    gan.G.save("../model/gan_gen.pkl")

