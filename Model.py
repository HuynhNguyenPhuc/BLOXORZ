from keras import Model
from keras.layers import Dense, Flatten
from keras.layers import BatchNormalization, Embedding
from keras.layers import LSTM

class LSTM_Model(Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun
    
    def __init__(self, input_size, num_features, units):
        super().__init__()
        
        # input_shape = (input_size, num_features)
        self.input_size = input_size
        self.num_features = num_features
        self.units = units

        self.embedding = Embedding(16, self.units, input_length = self.input_size, mask_zero = True)
        
        self.lstm1 = LSTM(
            512,
            input_shape = (self.units, self.num_features),
            return_sequences = True,
            recurrent_initializer='glorot_uniform'
        )
        
        self.lstm2 = LSTM(
            512,
            input_shape = (self.units, self.num_features),
            return_sequences = False,
            recurrent_initializer='glorot_uniform'
        )
        
        self.dense = Dense(self.num_features, name = 'dense')
        self.softmaxDense = Dense(self.num_features, activation="softmax", name = 'softmax-dense')
    
    def call(self, x):
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense(x)
        return x