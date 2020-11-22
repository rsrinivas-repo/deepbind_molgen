import pandas as pd
from utils import SmilesUtils
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda ,Dropout ,Reshape
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution2D , Convolution1D
from utils import  ConfigFile
import  os
import numpy as np
from keras.models import load_model
import keras
from keras import optimizers

latent_size = 50
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def cosine_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1, keepdims=True)

def sampler(args):
    mean, log_stddev = args

    std_norm = K.random_normal(shape=(K.shape(mean)[0], latent_size), mean=0, stddev=1)
    # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    return mean + K.exp(log_stddev) * std_norm

class RNNModelForSmiles(object):

    def __init__(self):

        self.datafile = ConfigFile.getProperty("implicit.smiles.data.file")
        self.charset_length=56
        self.max_length =120


    def buildModel(self):

        #print("Building model class")
        #strModelPath = os.path.join(ConfigFile.getProperty("models.dir"), "Implicit_to_Latent.h5")
        #loaded_model = load_model(strModelPath,
        #                        custom_objects={'cosine_distance': cosine_distance})
        inputLayer = Input(shape=(50,))



        #mean = Dense(latent_size)(inputLayer)
        #log_stddev = Dense(latent_size)(inputLayer)

        #latent_vector = Lambda(sampler)([mean, log_stddev])

        h = Dense(80, name='latent_input_11', activation='relu') (inputLayer)
        h = Dropout(0.2)(h)

        h = Dense(self.max_length, name='latent_input_12', activation='relu') (h)
        h = Dropout(0.2)(h)

        #h = Dense(self.max_length, name='latent_input', activation='relu') (inputLayer)
        #h = Reshape((12,10))(h)
        #h = Convolution1D(10, (9), activation='relu', name='conv_1', padding="same")(h)
        #h = Dropout(0.25)(h)
        #h = Convolution1D(6, (9), activation='relu', name='conv_2' , padding="same")(h)
        #h = Dropout(0.25)(h)
        #h = Convolution1D(3, (9), activation='relu', name='conv_3' , padding="same")(h)
        #h = Dropout(0.25)(h)

        #h = Flatten(name='flatten_1')(h)
        #h = Dense(60, name='latent_input_1', activation='relu')(h)
        h = Dense(self.max_length, name='latent_input_2', activation='relu')(h)

        h = RepeatVector(self.max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='gru_1')(h)
        h = GRU(501, return_sequences=True, name='gru_2')(h)
        h = GRU(501, return_sequences=True, name='gru_3')(h)
        outputLayer = TimeDistributed(Dense(self.charset_length, activation='softmax'), name='decoded_mean')(h)

        model = Model(inputLayer, outputLayer)
        #model = Model(inputs=[inputLayer.input], outputs=[h(inputLayer.output)])

        return  model


    def trainModel(self,molreg_arr,yPredArr):

        self.model = self.buildModel()

        sgd = optimizers.SGD(lr=.0001)
        self.model.compile(optimizer=sgd,
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=0.2,
                                      patience=3, min_lr=0.00001)

        print("fitting RNN model")
        self.history = self.model.fit(
                            molreg_arr,
                            yPredArr,
                            shuffle = True,
                            epochs = 100,
                            batch_size = 600,validation_split=.25 , callbacks=[reduce_lr]
                        )


    def saveModels(self):

        print("Saving model files")
        strModelPath =  os.path.join( ConfigFile.getProperty("models.dir") ,"Implicit_to_smiles_rnn.h5")
        strHistFilePath= os.path.join( ConfigFile.getProperty("models.dir") ,"Implicit_to_smiles_rnn_history.dict")
        self.model.save(strModelPath)
        with open(strHistFilePath, "w") as fp:
            fp.write(str(self.train_history.history))


def buildAndTrainRNNModel():

    rnnModel = RNNModelForSmiles()

    print("Inside main method TrainRNNModel")
    dfFile = pd.read_csv(rnnModel.datafile)

    print("Loaded data file")

    print ("Shape of implicit features file %s" % str(dfFile.shape))
    molreg_features = ["%d_mols" % i for i in range(0, 50)]
    molreg_arr = dfFile[molreg_features].values

    yPredArr = np.load(ConfigFile.getProperty("smile.onehot.file"))
    print ("Shape of the one hot encoded file %s"%str(yPredArr.shape))

    np.random.shuffle(molreg_arr)   #=molreg_arr[:20000,:]
    np.random.shuffle(yPredArr)     #=yPredArr[:20000,:,:]
    rnnModel.trainModel(molreg_arr,yPredArr)



if __name__ == '__main__':
    buildAndTrainRNNModel()
