from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import  Input, Dense, Activation, BatchNormalization, Conv2DTranspose, LeakyReLU, Conv2D
from keras.optimizers import Adam

class GAN(object):
    def __init__(self, in_shape, dense_dim, out_dim):
        self.shape = Input(shape=in_shape)
        self.dense_dim = dense_dim
        self.out_dim = out_dim
        pass

    def prepare_data(self):
        pass

    def generator(self, filters, k_size ):
        alpha=.2

        model = Sequential()

        model.add(Conv2D(filters*4, input_shape=self.shape,  kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        #model.add(Activation('relu'))
        model.add(LeakyReLU(alpha))

        model.add(Conv2D(filters*2, kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha))
        #model.add(Activation('relu'))

        model.add(Conv2D(filters, kernel_size=k_size, use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha))
        #model.add(Activation('relu'))

        model.add(Conv2DTranspose(filters/2, kernel_size=k_size, use_bias=False))
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
    def discriminator(self,filters, k_size):
        pass
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

        # 0 - 1 scale on label fake/real
        #model.add(LeakyReLu(alpha))
        model.add(Activation('softmax'))

        
        return model
        pass

if __name__ == "__main__":
    print("working")
    a = GAN(784, 12, 64 )
    gen = a.generator(16, (2,2));
    ge.summary()
    pass
