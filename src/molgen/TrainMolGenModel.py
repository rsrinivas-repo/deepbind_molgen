"""
This file contains source code to build and train a deep learning model to generate encoded smiles from
implicit ligand and target features

"""
import keras
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import numpy as np
from utils import ConfigFile as ConfigFile
from keras import optimizers
from utils import mVAE_helper
import  os
import  tensorflow as tf
import keras.backend as K



encoded_feature_size=None

class GenEncodedSMILES(object):

    def __init__(self):
        print("Initializing object for GenEncodedSMILES")
        self.constructModel()

    def constructModel(self):

        print("Size of latent dimensions %d"%encoded_feature_size)
        targetInput = Input(shape=(50,))
        #x1 = Dense(75, activation='linear')(targetInput)
        #x1 = Dense(100, activation='linear')(x1)

        molregnoInput = Input(shape=(50,))
        x2 = Dense(75, activation='linear')(molregnoInput)
        x2 = Dense(100, activation='linear')(x2)


        #mergedLayer = merge([x1, x2], concat_axis=1, mode="concat")
        #mergedLayer = keras.layers.Multiply()([x1,x2])
        #mergedLayer = keras.layers.Add()([x1, x2])
        ####merge function is depreceted - change it next revision

        x = Dense(100, activation='selu')(x2)
        x = Dense(150, activation='selu')(x)
        #x = Dense(195, activation='selu')(x)
        #x = Dense(210, activation='selu')(x)
        x = Dense(225, activation='selu')(x)
        #x = Dense(250, activation='selu')(x)
        #x = Dense(265, activation='selu')(x)
        #x = Dense(275, activation='selu')(x)
        outputLayer = Dense(encoded_feature_size, activation='selu')(x)

        self.model = Model([targetInput, molregnoInput], outputLayer)

        return  self.model



    @staticmethod
    def valid_predictions(y_pred):
        print("Finding number of valid predictions for array shaped %s"%str(y_pred.shape))

        countValids = 0
        listOutputs = []
        listInValid = list()

        fp = open("../data/InvalidSMILESfromSeqModels.csv","w")
        for i in range(0, y_pred.shape[0]):
            tArr = y_pred[i, :]

            decoded = mVAE_helper.isValidEncoding(tArr)

            if decoded!= None:
                listOutputs.append(decoded)

            else:
                line = mVAE_helper.getCommaSepString(tArr)
                fp.write(line)

        fp.close()
        return (listOutputs)



    def trainModel(self,targetArr , molregNoArr, latfeatureArr,epochs=5,batch_size=256,learningRate=.01):

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='mae', factor=0.2,
                                      patience=3, min_lr=0.00001)


        print("Training model")
        #optimizerFunction = optimizers.Adam(lr=learningRate)
        optimizerFunction = optimizers.Adagrad(lr=learningRate , decay=.0001)
        self.model.compile(optimizer=optimizerFunction, loss=euclidean_distance_loss ,
                      metrics=['mae', 'accuracy'])


        self.train_history = self.model.fit([targetArr, molregNoArr], latfeatureArr,
                                  epochs=epochs,
                                  batch_size=batch_size, callbacks=[reduce_lr],
                                  shuffle=False)


        #yPred = self.model.predict([targetArr, molregNoArr])

        #print("Number of valid predictions %d"% len(self.valid_predictions(yPred)))

        return  self.train_history


    def saveModel(self):

        print("Saving model files")
        strModelPath =  os.path.join( ConfigFile.getProperty("models.dir") ,"Implicit_to_Latent.h5")
        strHistFilePath= os.path.join( ConfigFile.getProperty("models.dir") ,"trainingHistory.dict")
        self.model.save(strModelPath)
        with open(strHistFilePath, "w") as fp:
            fp.write(str(self.train_history.history))

        return strModelPath


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def loadTrainingFile():
    dfAllData = pd.read_csv(ConfigFile.getProperty("implicit.data.file"))
    return dfAllData


def getTrainingData() :
    global encoded_feature_size
    encoded_feature_size=int(ConfigFile.getProperty("encoded.feature.size"))

    targetCols = ["%d_target"%i for i in range(0,50)]
    molregnoCols =  ["%d_mols"%i for i in range(0,50)]
    latentFeatCols = ["%d_latfeatures"%i for i in range(0,292)]


    dfTrainingData=loadTrainingFile()
    dfTrainingData=dfTrainingData.drop_duplicates("smiles")
    dfTrainingData = dfTrainingData.sample(50000)

    ## Extract numpy arrays with target , mol and encoded features
    targetArr = np.asarray(dfTrainingData.as_matrix(columns=targetCols))
    molregNoArr = np.asarray(dfTrainingData.as_matrix(columns=molregnoCols))
    latfeatureArr = np.asarray(dfTrainingData.as_matrix(columns=latentFeatCols))

    print("Shape of training file %s"%str(dfTrainingData.shape))

    print("Normalizing input")
    #targetArr =tf.keras.utils.normalize(targetArr)
    molregNoArr=tf.keras.utils.normalize(molregNoArr)
    latfeatureArr = keras.utils.normalize(latfeatureArr)
    print(np.mean(targetArr),np.mean(molregNoArr),np.mean(latfeatureArr),)
    return molregNoArr,targetArr,latfeatureArr

def trainSeqModel(epochs=5,batch_size=256,learningRate=.000001):

    print("start : Generate encoded smiles string from implicit fingerprints")

    molregNoArr, targetArr, latfeatureArr=getTrainingData()

    genEncoder = GenEncodedSMILES()

    print("Model Summary")
    print(genEncoder.model.summary())
    ## Train Model
    train_history = genEncoder.trainModel(targetArr, molregNoArr, latfeatureArr,epochs,batch_size,learningRate)

    # Save Model
    strModelPath = genEncoder.saveModel()
    print("end : Generate encoded smiles string from implicit fingerprints")

    return strModelPath


if __name__ == '__main__':

    trainSeqModel(epochs=2,batch_size=5)
