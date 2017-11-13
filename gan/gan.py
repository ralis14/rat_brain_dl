from __future__ import print_function
from keras.models import Model
from keras.layers import Sequential, Input, Dense, Activation, BatchNormalization, Conv2DTranspose, LeakyReLu, Conv2D
from keras.optimizers import Adam

class GAN(object):
    def __init__(self, in_shape, dense_dim, out_dim):
        self.shape = Input(shape=in_shape)
        self.dense_dim = dense_dim
        self.out_dim = out_dim
        pass
    def Generator(self, filters, k_size, ):
        model = Sequential()

        model.add(Conv2DTranspose(filters*4, kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters*2, kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters, kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2DTranspose())
        model.add(Activation('tanh'))
        return model
        #model = Dense(self.dense_dim)(self.shape)
        #model = Activation('tanh')(model)
        #model = Dense(self.out_dim, activation='tanh')(model)
        #out = Model(self.shape, model)
        #optimizer = Adam(self.lr)
        #out.compile(loss='binary_crossentropy', optimizer='adam')
        #return out, model 
        pass
    def Discriminator(self,filters, k_size):
        alpha = .2
        model = Sequential()

        model.add(Conv2D())
        model.add(LeakyReLu(alpha))
        
        model.add(Conv2D(filters, kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLu(alpha))

        model.add(Conv2D(filters*2, kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLu(alpha))

        model.add(Conv2D(filters*4, kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLu(alpha))

        
        return model
        pass

if __name__ == "__main__":
    print("working")
    pass
