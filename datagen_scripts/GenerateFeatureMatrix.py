import pandas as pd

import sys

print("Starting script to generate data")

molFP = pd.read_csv("../data/CompoundFingerPrints_LatentFectors_MultiBinary.csv")

targetFP = pd.read_csv("../data/TargetFingerPrints_LatentFectors_MultiBinary.csv")

assays = pd.read_csv("../data/AssayDataSetForDeeplearning.csv")

num_mols = 150000
print("Generating the features for %d molregnos "%num_mols)
trainingSetMols = assays.molregno.sample(n=num_mols)

with open("../data/trainingMols.txt","w") as fp:
    fp.write(str(trainingSetMols))
    
    
validationSetMols = assays[~(assays.molregno.isin(trainingSetMols))].molregno

with open("../data/validationMols.txt","w") as fp:
    fp.write(str(validationSetMols))
    
trainingSetAssays = assays[assays.molregno.isin(trainingSetMols)]

trainingSetAssays= trainingSetAssays[["molregno","tid"]]

trainingSetAssays =trainingSetAssays.merge(targetFP,on="tid")


trainingSetAssays= trainingSetAssays.merge(molFP,on="molregno",suffixes=('_target','_mols'))


### Load MVAE
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from mVAE.model import MoleculeVAE
from mVAE.utils import encode_smiles, decode_latent_molecule, interpolate, get_unique_mols
#import mVAE.mVAE
# number of dimensions to represent the molecules
# as the model was trained with this number, any operation made with the model must share the dimensions.
latent_dim = 292

# trained_model 0.99 validation accuracy
# trained with 80% of ALL chembl molecules, validated on the other 20.
trained_model = 'mVAE/chembl_23_model.h5'
charset_file = 'charset.json'

aspirin_smiles = 'CC(=O)Oc1ccccc1C(=O)O'


# load charset and model
with open('mVAE/charset.json', 'r') as outfile:
    charset = json.load(outfile)

model = MoleculeVAE()
model.load(charset, trained_model, latent_rep_size = latent_dim)

smilesFile = "../data/smilesCode.csv"

smilesDF = pd.read_csv(smilesFile,sep="\t",names=["molregno","smiles"])

trainSMILES = smilesDF[smilesDF.molregno.isin(trainingSetAssays.molregno)]




trainSMILES.reset_index(inplace=True)

if "index" in trainSMILES.columns:
    del trainSMILES["index"]
    
    
from rdkit import Chem
from rdkit import RDLogger

# remove warnings and errors from notebook (lots of them due non valid molecule generation)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

print("Size of training smiles datafrome %d"%trainSMILES.shape[0])

sys.stdout.write(".")
listImages = []
listLatentFeatures = list()
for i in range(0,trainSMILES.shape[0]):

    sys.stdout.write(".")    
    try:
    ##Reconstruct to check if this a valid mol
        smilesCode= trainSMILES.iloc[i]["smiles"]
        molregno= trainSMILES.iloc[i]["molregno"]
        latentFeatures = encode_smiles(smilesCode, model, charset)
        reconsMol = Chem.MolFromSmiles(decode_latent_molecule(latentFeatures, model, 
                                               charset, latent_dim))
        if reconsMol!=None:
            listImages.append(reconsMol)
            #trainSMILES.set_value(148,"encodedSMILES",latentFeatures,inplace=True)
            #trainSMILES.loc[i,"encodedSMILES"]=latentFeatures.reshape(latent_dim,) #
            tempList =list(latentFeatures.reshape(latent_dim,))
            tempList.append(smilesCode)
            tempList.append(molregno)
            listLatentFeatures.append(tempList)
    except:
        continue


print("")
print("Length of latent features %d"%len(listLatentFeatures))

colNames = ["%d_latfeatures"%i for i in range(0,latent_dim)]
colNames.append("smiles")
colNames.append("molregno")

dfLatentFeatures = pd.DataFrame(listLatentFeatures,columns=colNames)


print("shape of trainingset assays before merge %d"%trainingSetAssays.shape[0])
trainingSetAssays= trainingSetAssays.merge(dfLatentFeatures,on="molregno",suffixes=('_target','_mols'))

print("shape of trainingset assays after  merge %d"%trainingSetAssays.shape[0])


targetCols = ["%d_target"%i for i in range(0,50)]
molregNoCols = ["%d_mols"%i for i in range(0,50)]
latfeatCols = ["%d_latfeatures"%i for i in range(0,latent_dim)]

arrayData = trainingSetAssays.as_matrix(columns=[targetCols+molregNoCols+latfeatCols])

print("Shape of array data %d"%arrayData.shape[0])																

trainingSetAssays.to_csv("../data/FeaturesAsMatrixv1.ALL.csv")

print("Done.")


