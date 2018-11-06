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


class GenEncodedSMILES(object):

    def __init__(self):
        print("Initializing object for GenEncodedSMILES")
        self.constructModel()

    def constructModel(self):

        print("Size of latent dimensions %d"%encoded_feature_size)
        targetInput = Input(shape=(50,))
        x1 = Dense(75, activation='linear')(targetInput)
        x1 = Dense(100, activation='linear')(x1)

        molregnoInput = Input(shape=(50,))
        x2 = Dense(75, activation='linear')(molregnoInput)
        x2 = Dense(100, activation='linear')(x2)


        #mergedLayer = merge([x1, x2], concat_axis=1, mode="concat")
        mergedLayer = keras.layers.Concatenate()([x1,x2])
        #mergedLayer = keras.layers.Add()([x1, x2])
        ####merge function is depreceted - change it next revision

        x = Dense(150, activation='linear')(mergedLayer)
        x = Dense(175, activation='linear')(x)
        x = Dense(200, activation='linear')(x)
        x = Dense(250, activation='linear')(x)
        outputLayer = Dense(encoded_feature_size, activation='linear')(x)

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



    def trainModel(self,targetArr , molregNoArr, latfeatureArr,epochs=5,batch_size=256):

        print("Training model")
        optimizerFunction = optimizers.Adam(lr=.001)

        self.model.compile(optimizer=optimizerFunction, loss='mse' ,
                      metrics=['mae', 'accuracy'])


        self.train_history = self.model.fit([targetArr, molregNoArr], latfeatureArr,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  shuffle=False)
        yPred = self.model.predict([targetArr, molregNoArr])

        print("Number of valid predictions %d"% len(self.valid_predictions(yPred)))

        return  self.train_history


    def saveModel(self):

        print("Saving model files")
        self.model.save("../../model/Implicit_to_Latent.h5")
        with open("../../model/trainingHistory.dict", "w") as fp:
            fp.write(str(self.train_history.history))



def loadTrainingFile():
    dfAllData = pd.read_csv(ConfigFile.getProperty("implicit.data.file"))
    return dfAllData



if __name__ == '__main__':



    print("start : Generate encoded smiles string from implicit fingerprints")
    encoded_feature_size = int(ConfigFile.getProperty("encoded.feature.size"))

    targetCols = ["%d_target"%i for i in range(0,50)]
    molregnoCols =  ["%d_mols"%i for i in range(0,50)]
    latentFeatCols = ["%d_latfeatures"%i for i in range(0,292)]


    dfTrainingData=loadTrainingFile()

    #dfTrainingData = dfTrainingData.head(20)

    ## Extract numpy arrays with target , mol and encoded features
    targetArr = np.asarray(dfTrainingData.as_matrix(columns=targetCols))
    molregNoArr = np.asarray(dfTrainingData.as_matrix(columns=molregnoCols))
    latfeatureArr = np.asarray(dfTrainingData.as_matrix(columns=latentFeatCols))

    print("Shape of training file %s"%str(dfTrainingData.shape))

    genEncoder = GenEncodedSMILES()
    #train_history = genEncoder.trainModel(targetArr[10,:],molregNoArr[10,:],latfeatureArr[10,:],epochs=1,batch_size=2)

    train_history = genEncoder.trainModel(targetArr, molregNoArr, latfeatureArr,epochs=2,batch_size=5)
    """
    for i in range(0,4):
        print(mVAE_helper.isValidEncoding(latfeatureArr[i,:]))
        print(latfeatureArr[i,:10])
    """
    print("end : Generate encoded smiles string from implicit fingerprints")

