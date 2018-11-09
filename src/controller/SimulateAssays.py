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

def checkifValid(input):
    from utils import mVAE_helper

    decoded = mVAE_helper.isValidEncoding(input)
    if decoded is None :
        return  ("Invalid",1)
    return ("Valid",1)

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

    zipPath ="/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/src/molgen7.zip"
    sc = spark.sparkContext
    sc.addPyFile("/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/src/molgen7.zip")

    return  spark

def testConfiguration():
    featuresFile = pd.read_csv(ConfigFile.getProperty("implicit.data.file"))

    encodedColNames = ["%d_latfeatures" % i for i in range(0,
                                                           int(ConfigFile.getProperty("encoded.feature.size")))]

    arr = featuresFile[encodedColNames].values
    arr =arr[:40,:]

    print(arr.shape)
    #arr = np.random.rand(3,292)
    parallizeAndValidate(arr)

def parallizeAndValidate(arr):

    print("Parallize input array for Spark")
    spark = initSpark()
    rddArray = spark.sparkContext.parallelize(arr)

    print("Submitting job to Spark Mapper")
    mappedArr = rddArray.map(checkifValid)

    print("Submitting job to Spark reduceByKey")

    finalVals = mappedArr.reduceByKey(calcCounts).take(2)
    print(finalVals)

    return  finalVals

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

        genModel = TrainGANModel.trainGANModel()

        model = keras.models.load_model(genModel)

        invalidMols = np.load(ConfigFile.getProperty("descriminator.invalid.input"))

        print("Shape of test set %s"%str(invalidMols.shape))

        yPred = model.predict(invalidMols[:10,:])
        print("Shape of predicted output %s"%str(yPred.shape))

        parallizeAndValidate(yPred)