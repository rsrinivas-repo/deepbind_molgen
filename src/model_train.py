from  model import MolGen
import pandas as pd
import numpy as np 
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau , EarlyStopping


impl_molfeatures = ["%d_mols"%x for x in range(50)]
dfDataSet = pd.read_csv("../data/ImplicitFPs_550k.csv")


dfDataSet["smiles_len"] = dfDataSet.smiles.apply(lambda x : len(str(x)))

dfDataSet =dfDataSet[dfDataSet["smiles_len"]<120] 


train_arr_x = dfDataSet.sample(frac=.8)

val_arr_x = dfDataSet[~(dfDataSet.index.isin(train_arr_x.index))]

train_arr_x.shape,val_arr_x.shape




SMILES_CHARS = ["E", "(", ".", "0", "2", "4", "6", "8", "@", "B", "F", "H", "L", "N", "P", "R", "T", "V", "X", "Z", "\\", "b", "d", "l", "n", "p", "r", "t", "#", "%", ")", "+", "-", "/", "1", "3", "5", "7", "9", "=", "A", "C", "G", "I", "K", "M", "O", "S", "[", "]", "a", "c", "e", "g", "i", "o", "s", "u"]

smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))

def convertPredListToSMILES(ypred):
    return "".join([SMILES_CHARS[idx] for idx in ypred])

def convertPredtoList(ypred):
    ### Takes a 3 dim array 
    return ypred.argmax(axis=2)[0]


train_arr_y =  np.zeros(shape=(train_arr_x.shape[0],120,len(SMILES_CHARS)))
val_arr_y =  np.zeros(shape=(val_arr_x.shape[0],120,len(SMILES_CHARS)))

## setup train Y array 
for i in range(train_arr_x.shape[0]):
    smiles_string = train_arr_x.iloc[i]["smiles"]
    for j,char in enumerate(smiles_string):
        train_arr_y[i,j,smi2index[char]] = 1
    for k in range(j+1,120):
        train_arr_y[i,k,smi2index['E']]=1


## setup validation Y array 
for i in range(val_arr_y.shape[0]):
    smiles_string = val_arr_x.iloc[i]["smiles"]
    for j,char in enumerate(smiles_string):
        val_arr_y[i,j,smi2index[char]] = 1
    for k in range(j+1,120):
        val_arr_y[i,k,smi2index['E']]=1




checkpointer = ModelCheckpoint(filepath='saved_model.hdf5', verbose=1, save_best_only=True,monitor="loss")

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=3, min_lr=0.00001)

es = EarlyStopping(monitor='loss', min_delta=0, patience=6, verbose=0, mode='auto')

modelObject = MolGen()
modelObject.buildModel()
molgen_model = modelObject.compileModel()

history = molgen_model.fit(  #np.array(list_bit_string),
                            train_arr_x[impl_molfeatures].values,
                            train_arr_y,
                            shuffle = True,
                            epochs =2 ,
                            batch_size = 32,
                            validation_data = (val_arr_x[impl_molfeatures].values,val_arr_y)
                            ,callbacks=[checkpointer, reduce_lr,es  ]
                        )


