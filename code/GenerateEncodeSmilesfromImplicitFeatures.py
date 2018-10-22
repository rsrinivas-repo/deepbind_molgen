"""
This file contains source code to build and train a deep learning model to generate encoded smiles from
implicit ligand and target features

"""
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import merge
import pandas as pd
import numpy as np
from keras.models import Sequential
from ConfigFile import getProperty,reloadProperties
from keras import losses
from keras import optimizers


def loadTrainingFile():
    dfAllData = pd.read_csv(getProperty("implicit.data.file"))

    return dfAllData

def constructModel():

    print("Size of latent dimensions %d"%encoded_feature_size)
    targetInput = Input(shape=(50,))
    x1 = Dense(100, activation='linear')(targetInput)
    x1 = Dense(150, activation='linear')(x1)

    molregnoInput = Input(shape=(50,))
    x2 = Dense(100, activation='linear')(molregnoInput)
    x2 = Dense(150, activation='linear')(x2)

    #mergedLayer = merge([x1, x2], concat_axis=1, mode="concat")
    mergedLayer = keras.layers.Concatenate()([x1,x2])
    #mergedLayer = keras.layers.Add()([x1, x2])
    ####merge function is depreceted - change it next revision

    x = Dense(200, activation='linear')(mergedLayer)
    x = Dense(250, activation='linear')(x)
    outputLayer = Dense(encoded_feature_size, activation='linear')(x)

    model = Model([targetInput, molregnoInput], outputLayer)

    return  model



def trainModel(model, targetArr , molregNoArr, latfeatureArr):

    print("Training model")
    optimizerFunction = optimizers.Adam(lr=.0001)
    model.compile(optimizer=optimizerFunction, loss='mse' ,
                  metrics=['mae', 'acc'])
    train_history = model.fit([targetArr, molregNoArr], latfeatureArr,
                              epochs=100,
                              batch_size=256,
                              shuffle=True)

    return  train_history




if __name__ == '__main__':



    print("start : Generate encoded smiles string from implicit fingerprints")
    encoded_feature_size = int(getProperty("encoded.feature.size"))

    model = constructModel()

    targetCols = ["%d_target"%i for i in range(0,50)]
    molregnoCols =  ["%d_mols"%i for i in range(0,50)]
    latentFeatCols = ["%d_latfeatures"%i for i in range(0,292)]


    dfTrainingData=loadTrainingFile()
    ## Extract numpy arrays with target , mol and encoded features
    targetArr = np.asarray(dfTrainingData.as_matrix(columns=targetCols))
    molregNoArr = np.asarray(dfTrainingData.as_matrix(columns=molregnoCols))
    latfeatureArr = np.asarray(dfTrainingData.as_matrix(columns=latentFeatCols))

    print("Shape of training file %s"%str(dfTrainingData.shape))

    train_history = trainModel(model,targetArr,molregNoArr,latfeatureArr)

    print("end : Generate encoded smiles string from implicit fingerprints")

    model.save("../model/Implicit_to_Latent.model")
    with open("../model/trainingHistory.dict","w") as fp:
        fp.write(str(train_history.history))




