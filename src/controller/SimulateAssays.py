"""

This program invokes the keras model for training a Seq Model for generating encoded smiles and
validates the output to check the number of valid ones.

"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf,SQLContext
from utils import  ConfigFile
import  numpy as np
import os
import  pandas as pd
import argparse
from functools import partial
import json
import pickle
from molgen.SmilesDecoder import  SmilesDecoder


bVar=None

##This is the map method
def checkifValid(input,bVar):
    from utils import mVAE_helper
    from molgen.SmilesDecoder import  SmilesDecoder

    print("Inside Mapper , bVar is %s"%str(bVar))
    #print("Inside Mapper . Type of broadcast variable is %s "%str(isinstance(bVar.value,MoleculeVAE)))
    decoded = SmilesDecoder().isValidEncoding(bVar.value,input)


    if decoded is None :
        return  ("Invalid",1)
    return ("Valid",1)

##This is the reduce method
def calcCounts(a,b):
    return (a+b)

def initSpark():
    APP_NAME = "deepbind"
    spark_host = ConfigFile.getProperty("spark.host")
    spark_master = ConfigFile.getProperty("spark.master")
    conf = SparkConf().setAppName(APP_NAME)
    conf.set("spark.driver.host",spark_host)
    conf.set("spark.rpc.message.maxSize","1024")
    conf.setExecutorEnv("PATH",os.environ["PATH"])
    spark = SparkSession.builder.config(conf=conf).master(spark_master).getOrCreate()

    zipPath ="/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/src/molgen10.zip"
    sc = spark.sparkContext
    sc.addPyFile(zipPath)

    return  spark

def parallizeAndValidate(arr):

    print("Parallize input array for Spark")
    spark = initSpark()
    rddArray = spark.sparkContext.parallelize(arr)


    ## Init the mVAE model object and pass it as broadcast variable
    print("Loading decoder object")

    decoderObj = SmilesDecoder().initDecoderModel()

    print("Done initializing decoder object")


    sc = spark.sparkContext
    global  bVar
    bVar = sc.broadcast(decoderObj)

    print("Submitting job to Spark Mapper")
    mappedArr = rddArray.map(partial(checkifValid,bVar=bVar))

    print("Submitting job to Spark reduceByKey")

    finalVals = mappedArr.reduceByKey(calcCounts).take(2)
    print(finalVals)

    return  finalVals


##Utility method to test out all configurations
def testConfiguration():
    featuresFile = pd.read_csv(ConfigFile.getProperty("implicit.data.file"))

    encodedColNames = ["%d_latfeatures" % i for i in range(0,
                                                           int(ConfigFile.getProperty("encoded.feature.size")))]

    arr = featuresFile[encodedColNames].values
    arr =arr[:1000,:]

    print(arr.shape)

    parallizeAndValidate(arr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Usage --runmode test or --runmode genFromImplicit or --runmode genFromGAN')

    parser.add_argument('--runmode', help="Enter one a runmode to proceed"
                                           ,choices=['test','genFromImplicit','genFromGAN'],required=True)

    args = parser.parse_args()



    if args.runmode=="test" :

    ## This function tests spark configurations.
        testConfiguration()

    elif args.runmode=="genFromImplicit" :

        ##Train SeqModel and perform validation on the output
        from molgen import TrainMolGenModel
        import  keras

        strFile=TrainMolGenModel.trainSeqModel()

        print("The Model saved @ %s"%strFile)

        molregNoArr, targetArr, latfeatureArr = TrainMolGenModel.getTrainingData()

        model = keras.models.load_model(strFile)
        yPred = model.predict([targetArr, molregNoArr])

        print("Shape of predicted file " + str(yPred.shape))
        parallizeAndValidate(yPred)

        # print("Number of valid predictions %d"% len(self.valid_predictions(yPred)))

    elif args.runmode=="genFromGAN":

        from molgen import TrainGANModel
        import keras

        print("Train GAN Model and validate")

        # genModel = TrainGANModel.trainGANModel()

        # model = keras.models.load_model(genModel)

        model = keras.models.load_model("/users/rsrinivas/deepbind/paper2/deepbind_molgen/model/gan_gen.h5")
        invalidMols = np.load(ConfigFile.getProperty("descriminator.invalid.input"))

        print("Shape of test set %s" % str(invalidMols.shape))

        yPred = model.predict(invalidMols[:1000, :])
        print("Shape of predicted output %s" % str(yPred.shape))

        parallizeAndValidate(yPred)
