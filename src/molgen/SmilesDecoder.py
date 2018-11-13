from utils import  ConfigFile
from mVAE.model import MoleculeVAE
import json
import pickle
from mVAE.utils import decode_smiles_from_indexes
from rdkit import Chem
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras_pickle_wrapper import KerasPickleWrapper
import pandas as pd


class SmilesDecoder(object):
    """
    This class creates a SMILEs decoder object that converts the 292 bit vector representation to a valid SMILES string.
    The object is "pickle"-able and can be used as a broadcast variable for spark executors.
    """
    def __init__(self):

        charset_file = ConfigFile.getProperty("charset.file")
        with open(charset_file, 'r') as outfile:
            self.charset = json.load(outfile)

        self.latent_rep_size = int(ConfigFile.getProperty("encoded.feature.size"))
        self.max_length = 120
        self.charset_length = len(self.charset)


    def initDecoderModel(self):
        """
        Initialize SmilesDecoder Object  and load up the decder weghts from file
        :return: Pickleable decoder model
        """
        encoded_input = Input(shape=(292,))
        self.decoderModel = Model(
            encoded_input,
            MoleculeVAE()._buildDecoder(
                encoded_input,
                self.latent_rep_size,
                self.max_length,
                self.charset_length
            ))

        print("Loading up weights into the decoder model")
        self.decoderModel.load_weights(ConfigFile.getProperty("decoder.weights.file"))
        print("Done loading up weights into the decoder model")

        print("Creating pickle-able model")
        self.decoderModel = KerasPickleWrapper(self.decoderModel)
        return self.decoderModel



    def isValidEncoding(self,modelObject,inputArr):

        tArr = modelObject().predict(inputArr.reshape(1, self.latent_rep_size)).argmax(axis=2)[0]
        smiles = decode_smiles_from_indexes(tArr, self.charset)
        mol = Chem.MolFromSmiles(smiles)

        if mol :
            return smiles

        return  None

if __name__ == '__main__':

    ## THis contains sample code to test out this class.


    ## loadup the decoder model and predict
    decoderObj = SmilesDecoder().initDecoderModel()
    featuresFile = pd.read_csv(ConfigFile.getProperty("implicit.data.file"))

    encodedColNames = ["%d_latfeatures" % i for i in range(0,
                                                           int(ConfigFile.getProperty("encoded.feature.size")))]

    print("Begin sample predictions")
    for i in range(0, 10):
        tempArr = featuresFile.loc[i, encodedColNames]
        print(SmilesDecoder().isValidEncoding(decoderObj,tempArr))


    ##pickle model and load it back up and predict
    with open("/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/sample.pkl","w") as fp:
        pickle.dump(decoderObj,fp)

    with open("/Users/raghuramsrinivas/localdrive/education/deepbind/paper2/sample.pkl") as fp:
        savedModel = pickle.load(fp)


    print("Begin sample predictions")
    for i in range(0, 10):
        tempArr = featuresFile.loc[i, encodedColNames]
        print(SmilesDecoder().isValidEncoding(savedModel,tempArr))




