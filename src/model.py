import keras
from keras import backend as K
from keras import objectives
from keras.models import Model ,Sequential
from keras.layers import Input, Dense, Lambda ,Dropout ,Reshape,Bidirectional
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution2D , Convolution1D
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import regularizers



class MolGen:
    def __init__(self, sampling_stddev=.01):
        self.stddev = sampling_stddev

    def sampler(self,args):
        mean, log_stddev = args
        #global  latent_size
        std_norm = K.random_normal(shape=(K.shape(mean)[0], 50), mean=0, stddev=self.stddev)  

        return mean + K.exp(log_stddev) * std_norm

    def _buildGRULayers(self,latent_rep_size=292, max_length=120, charset_length=58):
        inputLayer = keras.layers.Input(shape=(292,))
        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(inputLayer)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)
        op =  TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

        return Model(inputLayer,op)

    
    def _buildDenseLayers(self):
        inputLayer = keras.layers.Input(shape=(50,))
        mean = Dense(50)(inputLayer)
        stddev = Dense(50)(inputLayer)
        sampled_data = Lambda(self.sampler)([mean, stddev])

        charset_length = 58

        bn1 = keras.layers.Dense(100,activation="relu"   )(sampled_data)

        bn1 = keras.layers.Dense(150,activation="relu"   )(bn1)
        bn1 = keras.layers.Dense(200,activation="relu"   )(bn1)

        bn1 = keras.layers.Dense(250,activation="relu")(bn1)

        outputDense = keras.layers.Dense(292,activation="relu" )(bn1)
        
        return inputLayer , outputDense
    
    def buildModel(self):
        
        inputLayer , outputDense = self._buildDenseLayers()
        gruDecoder =self._buildGRULayers()
        gruDecoder.Trainable=True
        self.fused_model = keras.models.Model(inputLayer,gruDecoder(outputDense))
        
        return self.fused_model
        
    def compileModel(self):
        opt = optimizers.RMSprop()  

        self.fused_model.compile(loss=  'categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])


        return self.fused_model
	