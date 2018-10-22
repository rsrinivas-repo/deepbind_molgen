from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import merge
import pandas as pd
import numpy as np
from keras.models import Sequential
import  mVAE_helper
#reload(mVAE_helper)
import  ConfigFile

import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from mVAE.model import MoleculeVAE
from mVAE.utils import encode_smiles, decode_latent_molecule, interpolate, get_unique_mols

from rdkit import Chem
from rdkit import RDLogger
import  sys


# remove warnings and errors from notebook (lots of them due non valid molecule generation)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


ConfigFile.reloadProperties()

dfAllData = pd.read_csv(ConfigFile.getProperty("implicit.data.file"))

print("Shape of datafile %s"%str(dfAllData.shape))

dfAllData.columns



encodedFeatures = ["%d_latfeatures"%i for i in range(0,int(ConfigFile.getProperty("encoded.feature.size")))]

def getDFWithEncodedFeatures():
    listFeatures = encodedFeatures+["smiles"]

    return  dfAllData[listFeatures]


dfEncodedFeatures = getDFWithEncodedFeatures()

encodedFeaturesArray = dfEncodedFeatures[encodedFeatures].as_matrix()

#### Load the VAE model and generate real and fake molecules

modelVae = mVAE_helper.loadmVAEModel()

with open(mVAE_helper.charset_file, 'r') as outfile:
    charset = json.load(outfile)

reconstructed_molecule = decode_latent_molecule(encodedFeaturesArray[1,:], modelVae,
                                               charset, mVAE_helper.latent_dim)


listValidMols=list()
listInvalidMols=list()

print("Generate valid and invalid encoded vectors for GAN")
invalidCnts=0
for i in range(0,encodedFeaturesArray.shape[0]):
#for i in range(0,5):
    sys.stdout.write(".")
    sys.stdout.flush()
    reconstructed_molecule = decode_latent_molecule(encodedFeaturesArray[i, :], modelVae,
                                                    charset, mVAE_helper.latent_dim)

    listValidMols.append(encodedFeaturesArray[i,:])
    invalidCnts=0
    ## Generate encoded values at random and save them as valid or invalid for GANs

    stdev = 0.1
    mvaeCount = int(ConfigFile.getProperty("mvae.molgen.count"))
    latent_mols = stdev * np.random.randn(mvaeCount, mVAE_helper.latent_dim) + encodedFeaturesArray[i, :]

    decoded_molecules = []
    for lm in latent_mols:
        smiles=decode_latent_molecule(lm, modelVae,
                                       charset, mVAE_helper.latent_dim)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                listValidMols.append(lm)
            else:
                if invalidCnts<30:
                    listInvalidMols.append(lm)
                    invalidCnts+=1
        except:
            continue

validMolsArray = np.array(listValidMols)
print("# of valid mols generated  %d"%validMolsArray.shape[0])
np.save("../data/gan_data_valid",validMolsArray,allow_pickle=False)

invalidMolsArray = np.array(listInvalidMols)
print("# of valid mols generated  %d"%invalidMolsArray.shape[0])
np.save("../data/gan_data_invalid",invalidMolsArray,allow_pickle=False)
