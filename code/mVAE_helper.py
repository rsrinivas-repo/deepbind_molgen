import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from mVAE.model import MoleculeVAE
from mVAE.utils import encode_smiles, decode_latent_molecule, interpolate, get_unique_mols
import  ConfigFile


# number of dimensions to represent the molecules
# as the model was trained with this number, any operation made with the model must share the dimensions.
latent_dim = int(ConfigFile.getProperty("encoded.feature.size"))

#ConfigFile.reloadProperties()

# trained_model 0.99 validation accuracy
# trained with 80% of ALL chembl molecules, validated on the other 20.
trained_model = ConfigFile.getProperty("mvae.trained.model")
charset_file = ConfigFile.getProperty("charset.file")

global charset

# load charset and model
def loadmVAEModel():

    print("Loading Autoencoder Model to encode smiles into numeric vectors")
    with open(charset_file, 'r') as outfile:
        charset = json.load(outfile)

    modelVae = MoleculeVAE()
    modelVae.load(charset, trained_model, latent_rep_size = latent_dim)
    print("Successfully loaded Autoencoder model")
    return  modelVae


