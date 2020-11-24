from rdkit import Chem
from rdkit import RDLogger
from model import MolGen
import pandas as pd
import numpy as np





df = pd.read_csv("../data/Implicit_Fps50k.csv")
df = df[df.TotalPosAffinities>10]


df = df.drop_duplicates(subset="smiles")
df.dropna(inplace=True)
df.reset_index(inplace=True)





modelObj = MolGen(sampling_stddev=.0001)
model = modelObj.buildModel()

model_location = "../model/saved_model.hdf5"
model.load_weights(model_location)


RDLogger.DisableLog('rdApp.*')

def isValidEncoding(smiles):
    """
    :param encodingArr: Encoding array to convert to SMILES string
    :return:  returns valid Smiles string or None is the input array is an invalid encoding
    """
    #return  1
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol :
            return True
    except Exception as e :
            print(e)
            return False
    return False


def convertPredListToSMILES(ypred):
    return "".join([SMILES_CHARS[idx] for idx in ypred])

def convertPredtoList(ypred):
    ### Takes a 3 dim array 
    return ypred.argmax(axis=2)[0]



impl_molfeatures = ["%d_mols"%x for x in range(50)]

#from utils import ConfigFile, mVAE_helper
SMILES_CHARS = [" ", "(", ".", "0", "2", "4", "6", "8", "@", "B", "F", "H", "L", "N", "P", "R", "T", "V", "X", "Z", "\\", "b", "d", "l", "n", "p", "r", "t", "#", "%", ")", "+", "-", "/", "1", "3", "5", "7", "9", "=", "A", "C", "G", "I", "K", "M", "O", "S", "[", "]", "a", "c", "e", "g", "i", "o", "s", "u"]



list_Implicit_fps=list()
list_class_information = list()
smiles_set = set()
list_smiles = list()

counter = 0
i=  20
for i in  range(10):   #range(df.shape[0]):

    print("************************\n Orignial SMILES \n"+df.iloc[i]["smiles"])

    x_arr = df.iloc[i][impl_molfeatures]
    x_arr = x_arr.values.reshape(1,50)

    list_Implicit_fps.append(x_arr)
    list_class_information.append(1)

    for j in range(50):

        noise = np.random.normal(0,.01,(1,50))

        new_arr = x_arr + noise
        ypred = model.predict(new_arr)

        pred_smiles = convertPredListToSMILES(convertPredtoList(ypred))

        print(pred_smiles)
        temp_dict = dict()
        temp_dict["orig_smiles"] = df.iloc[i]["smiles"]
        temp_dict["orig_molregno"] = df.iloc[i]["molregno"]
        temp_dict["pred__smiles"] = pred_smiles
        temp_dict["flag"] = "Invalid"

        list_Implicit_fps.append(new_arr)

        if (isValidEncoding(pred_smiles)):
             print("Valid")
             temp_dict["flag"] = "Valid"
             smiles_set.add(pred_smiles)
             list_smiles.append(temp_dict)
             list_class_information.append(i)
        else:
             list_class_information.append(0)


print("Number of Unique Ligands Generated %s "%len(smiles_set))
                          
