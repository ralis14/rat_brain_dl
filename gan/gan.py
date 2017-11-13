from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import Adam

class GAN(object):
    def __init__(self, in_shape, dense_dim, out_dim):
        self.shape = Input(shape=in_shape)
        self.dense_dim = dense_dim
        self.out_dim = out_dim
        pass
    def Generator(self):
        model = Dense(self.dense_dim)(self.shape)
        model = Activation('tanh')(model)
        model = Dense(self.out_dim, activation='tanh')(model)
        out = Model(self.shape, model)
        #optimizer = Adam(self.lr)
        out.compile(loss='binary_crossentropy', optimizer='adam')
        return out, model 
        pass
    def Discriminator(self):
        pass

if __name__ == "__main__":
    print("working")
    pass
