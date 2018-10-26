import pandas as pd
import numpy as np
from  keras.models import load_model


model = load_model("../model/gan_gen.pkl")

noise = np.random.normal(0, 1, (5, 292))

out = model.predict(noise) 

print(out)


import json
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from mVAE.model import MoleculeVAE
from mVAE.utils import encode_smiles, decode_latent_molecule, interpolate, get_unique_mols

from rdkit import Chem
from rdkit import RDLogger
import mVAE_helper


modelVae = mVAE_helper.loadmVAEModel()

with open(mVAE_helper.charset_file, 'r') as outfile:
    charset = json.load(outfile)

for i in range(0, out.shape[0]):
                                                    
	smiles=decode_latent_molecule(out[i,:], modelVae,
                                       charset, mVAE_helper.latent_dim)
                                       
	mol = Chem.MolFromSmiles(smiles)

	print(mol)

print("Done.")
