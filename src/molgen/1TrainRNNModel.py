import pandas as pd
from utils import SmilesUtils
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from utils import  ConfigFile
import  os
import numpy as np
import tensorflow as tf 

class RNNModelForSmiles(object):

    def __init__(self):

        self.datafile = ConfigFile.getProperty("implicit.smiles.data.file")
        self.charset_length=56
        self.max_length =120



    def buildModel(self):

        print("Building model class")
        inputLayer = Input(shape=(50,))
        h = Dense(self.max_length, name='latent_input', activation='relu')(inputLayer)
        h = RepeatVector(self.max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='gru_1')(h)
        h = GRU(501, return_sequences=True, name='gru_2')(h)
        h = GRU(501, return_sequences=True, name='gru_3')(h)
        outputLayer = TimeDistributed(Dense(self.charset_length, activation='softmax'), name='decoded_mean')(h)

        model = Model(inputLayer, outputLayer)

        return  model


    def trainModel(self,molreg_arr,yPredArr):

        self.model = self.buildModel()

        self.model.compile(optimizer='Adam',
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])


        print("fitting RNN model")
        self.history = self.model.fit(
                            molreg_arr,
                            yPredArr,
                            shuffle = True,
                            epochs = 50,
                            batch_size = 600,validation_split=.25
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
    molreg_features = ["%d_mols" % i for i in range(0, 50)]
    molreg_arr = dfFile[molreg_features].values

    molreg_arr=tf.keras.utils.normalize(molreg_arr)
    yPredArr = np.load(ConfigFile.getProperty("smile.onehot.file"))
    print ("Shape of the one hot encoded file %s"%str(yPredArr.shape))

    #molreg_arr=molreg_arr[:15000,:]
    #yPredArr=yPredArr[:15000,:,:]
    rnnModel.trainModel(molreg_arr,yPredArr)



if __name__ == '__main__':
    buildAndTrainRNNModel()
