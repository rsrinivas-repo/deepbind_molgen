"""

This program invokes the keras model for training a Seq Model for generating encoded smiles and
validates the output to check the number of valid ones.

"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf, SQLContext
from utils import ConfigFile
import numpy as np
import os
import pandas as pd
import argparse
from functools import partial
import json
import pickle
from molgen.SmilesDecoder import SmilesDecoder
import tensorflow as tf
import keras.backend as K



bVar = None

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


##This is the map method
def checkifValid(input, bVar):
    from utils import mVAE_helper
    from molgen.SmilesDecoder import SmilesDecoder

    print("Inside Mapper , bVar is %s" % str(bVar))
    # print("Inside Mapper . Type of broadcast variable is %s "%str(isinstance(bVar.value,MoleculeVAE)))
    decoded = SmilesDecoder().isValidEncoding(bVar.value, input)

    if decoded is None:
        return ("Invalid", 1)
    return ("Valid", 1)


##This is the reduce method
def calcCounts(a, b):
    return (a + b)


def initSpark():
    APP_NAME = "deepbind"
    spark_host = ConfigFile.getProperty("spark.host")
    spark_master = ConfigFile.getProperty("spark.master")
    conf = SparkConf().setAppName(APP_NAME)
    conf.set("spark.driver.host", spark_host)
    conf.set("spark.rpc.message.maxSize", "1024")
    conf.setExecutorEnv("PATH", os.environ["PATH"])

    conf.setExecutorEnv("PATH", os.environ["PATH"])

    envInfo = os.getenv("deepbind_env")
    conf.setExecutorEnv("deepbind_env", envInfo)

    spark = SparkSession.builder.config(conf=conf).master(spark_master).getOrCreate()

    zipPath = ConfigFile.getProperty("spark.workers.pyfile")

    sc = spark.sparkContext
    sc.addPyFile(zipPath)

    return spark


def parallizeAndValidate(arr):
    print("Parallize input array for Spark")
    spark = initSpark()
    rddArray = spark.sparkContext.parallelize(arr, numSlices=28)

    ## Init the mVAE model object and pass it as broadcast variable
    print("Loading decoder object")

    decoderObj = SmilesDecoder().initDecoderModel()

    print("Done initializing decoder object")

    sc = spark.sparkContext
    global bVar
    bVar = sc.broadcast(decoderObj)

    print("Submitting job to Spark Mapper")
    mappedArr = rddArray.map(partial(checkifValid, bVar=bVar))

    print("Submitting job to Spark reduceByKey")

    finalVals = mappedArr.reduceByKey(calcCounts).take(2)
    print(finalVals)

    return finalVals


##Utility method to test out all configurations
def testConfiguration():
    featuresFile = pd.read_csv(ConfigFile.getProperty("implicit.data.file"))

    encodedColNames = ["%d_latfeatures" % i for i in range(0,
                                                           int(ConfigFile.getProperty("encoded.feature.size")))]

    arr = featuresFile[encodedColNames].values
    arr = arr[:1000, :]

    print(arr.shape)

    parallizeAndValidate(arr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Usage --runmode test or --runmode genFromImplicit or --runmode genFromGAN')

    parser.add_argument('--runmode', help="Enter one a runmode to proceed"
                        , choices=['test', 'genFromImplicit', 'genFromGAN'], required=True)

    args = parser.parse_args()

    if args.runmode == "test":

        ## This function tests spark configurations.
        testConfiguration()

    elif args.runmode == "genFromImplicit":

        ##Train SeqModel and perform validation on the output
        from molgen import TrainMolGenModel
        import keras

        #strFile=TrainMolGenModel.trainSeqModel(epochs=100,batch_size=256,learningRate=.001)

        strFile = os.path.join(ConfigFile.getProperty("models.dir"), "Implicit_to_Latent.h5")

        print("Saved model  @ %s" % strFile)

        molregNoArr, targetArr, latfeatureArr = TrainMolGenModel.getTrainingData()

        #targetArr = tf.keras.utils.normalize(targetArr)
        #molregNoArr = tf.keras.utils.normalize(molregNoArr)


        model = keras.models.load_model(strFile ,  custom_objects={'cosine_distance': cosine_distance})
        yPred = model.predict(molregNoArr)

        print("Shape of predicted file " + str(yPred.shape))
        import random

        random_indexes = random.sample(range(0, yPred.shape[0]), 100)

        parallizeAndValidate(testArr)

        #YHatArr = latfeatureArr[random_indexes, :]
        #testArr = yPred[random_indexes, :]

        #print("Shape of test Arr %s"%str(testArr.shape))

        #listNewVectors = []
        #for i in range(0,10):
        #    stdev = 0.9
        #    mvaeCount = 1000    #int(ConfigFile.getProperty("mvae.molgen.count"))
        #    latentSize=int(ConfigFile.getProperty("encoded.feature.size"))
        #    latent_mols = stdev * np.random.randn(mvaeCount,latentSize) + testArr[i, :]
        #    print("Shape of latent_mols Arr %s" % str(latent_mols.shape))
        #    listNewVectors.append(latent_mols)
        #    parallizeAndValidate(latent_mols)

        #newTestArr = np.array(listNewVectors)
        #print("Shape of new test array %s"%str(newTestArr.shape))


# print("Number of valid predictions %d"% len(self.valid_predictions(yPred)))

    elif args.runmode == "genFromGAN":

        from molgen import TrainGANModel
        import keras

        print("Train GAN Model and validate")

        strFile = TrainGANModel.trainGANModel()

        strFile = os.path.join(ConfigFile.getProperty("models.dir"), "gan_gen.h5")

        model = keras.models.load_model(strFile)

        invalidMols = np.load(ConfigFile.getProperty("descriminator.invalid.input"))

        print("Shape of test set %s" % str(invalidMols.shape))

        yPred = model.predict(invalidMols)
        print("Shape of predicted output %s" % str(yPred.shape))

        parallizeAndValidate(yPred)
