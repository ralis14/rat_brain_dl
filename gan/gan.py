from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import  Input, Flatten, UpSampling2D, Dense, Activation, BatchNormalization, Conv2DTranspose, LeakyReLU, Conv2D, Dropout, Reshape
from keras.optimizers import Adam, RMSprop
from tqdm import tqdm
import numpy as np

class Gen(object):
    """
    dimensions and channels are used to compute the transpose convolutions in the generator,
    since they cna change it is better to ask for them at initialization and store them
    """
    def __init__(self, dim1, dim2, channels, drop):
        self.dim1 = dim1//4
        self.dim2 = dim2//4
        self.chan = channels
        self.start_depth = 256
        self.drop = drop
    def main(self):
        temp = self.dim1*self.dim2*self.start_depth
        model = Sequential()
        model.add(Dense(temp, input_dim=100))
        model.add(BatchNormalization(momentum=.9))
        model.add(Activation('relu'))
        model.add(Reshape((self.dim1, self.dim2, self.start_depth)))
        model.add(Dropout(self.drop))
        #shape of model should be (dim1//4, dim2//4, 256)

        model.add(UpSampling2D())
        model.add(Conv2DTranspose(self.start_depth//2, 5, padding='same'))
        model.add(BatchNormalization(momentum=.9))
        model.add(Activation('relu'))
        #after sampling dim=(dim//4)*2 or dim/2
        #after con2DTranspose channels become self.start_depth//4
        #new dimentions are (self.dim1//2, self.dim2//2, 64 )
        
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(self.start_depth//4, 5, padding='same'))
        model.add(BatchNormalization(momentum=.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(self.start_depth//8, 5, padding='same'))
        model.add(BatchNormalization(momentum=.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(self.chan, 5, padding='same'))
        model.add(Activation('sigmoid'))

        return model
class Disc(object):
    def __init__(self, alpha, dropout, input_shape=()):
        self.start_depth = 32
        assert dropout < 1, "Dropout must be less than 1"
        assert input_shape != (), 'input shape must be 3 values "(dimension1, dimension2, channels): {}"'.format(input_shape)
        self.input_shape = input_shape
        self.alpha = alpha
        self.dropout = dropout

    def main(self, optimize=RMSprop(lr=.0008, clipvalue=1.0, decay=6e-8)):
        model = Sequential()
        model.add(Conv2D(self.start_depth, 5, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(self.alpha))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(self.start_depth*2, 5, strides=2, padding='same'))
        model.add(LeakyReLU(self.alpha))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(self.start_depth*4, 5, strides=2, padding='same'))
        model.add(LeakyReLU(self.alpha))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(self.start_depth*8, 5, strides=1, padding='same'))
        model.add(LeakyReLU(self.alpha))
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=optimize, metics=['accuracy'])

        return model

class GAN(object):
    def __init__(self, gen, disc, batch_size=32):
        self.G = gen
        self.D = disc
        self.batch_size = batch_size

    """
    train_mode is a function to be able to freeze the weights on a model, 
    usually this is done to  a discriminator while generator is being trained
    """
    def train_mode(self, model, mode=False):
        model.trainable = mode
        for layer in model.layers:
            layer.trainable = mode
    def main(self, optimize=RMSprop(lr=.0004, clipvalue=1.0, decay=3e-8)):
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        model.compile(loss='binary_crossentropy', optimizer=optimize, metrics=['accuracy'])
        self.gan = model
        return self.gan

    #TODO return a subset, of sample subset of data
    def sample_data(self, num_samples=100):
        from keras.datasets import mnist
        (xtrain, ytrain),(xtest, ytest) = mnist.load_data()
        data = xtrain[:1000]
        data = np.reshape(data,(data.shape[0], 28, 28, 1))
        return data
    def s_data_and_gen(self, noise_dim=100, num_samples=100):
        x_temp = self.sample_data(num_samples=num_samples)
        x_temp_noise = np.random.uniform(0, 1, size=[num_samples, noise_dim])
        noise_pred = self.G.predict(x_temp_noise)

        print(x_temp.shape)
        print(noise_pred.shape)

        data = np.concatenate((x_temp, noise_pred))
        labels = np.zeros((2*num_samples, 2))
        labels[:num_samples, 1] = 1
        labels[num_samples, 0] = 1
        return data, labels

    """
    pretrain is a method to pretrain your discriminator,
    using real and fake data, generated images from generator and correctly labled data
    """
    def pre_train(self, model, data, labels, b_size=32):
        self.train_mode(model, mode=True)
        model.fit(data, labels, epochs=1, batch_size=b_size)

    """
    sample noise just takes in the gen input dimension or default of 100, and the number of actual samples of data you are using
    this will craete data and labels that will be used to be ran through the generator.
    """
    def sample_noise(self, noise_dim=100, num_samples=100):
        data = np.random.uniform(0, 1, size=[num_samples, noise_dim])
        labels = np.zeros((num_samples, 2))
        labels[:,1] = 1
        return data, labels

    """
    Alternate the training of the Discriminator and the gan with the frozen generator weights.
    This way both are trained but sequentially learning from each other
    """
    def train(self, epochs=400, samples=100, gen_input_dimension=100, out_verbose=True, progress_freq=50):
        disc_loss=[]
        gen_loss=[]
        loop = range(epochs)
        if out_verbose:
            loop = tqdm(loop)
        for epoch in loop:
            #train disc
            data, labels = self.s_data_and_gen()
            self.train_mode(self.D, mode=True)
            disc_loss.append(self.D.train_on_batch(data, labels))

            #train gan
            data, labels = self.sample_noise()
            self.train_mode(self.D, mode=False)
            gen_loss.append(self.gan.train_on_batch(data, labels))

            if out_verbose and (epoch+1) % progress_freq == 0:
                print('Epoch number {}: Gen loss: {}, Disc loss: {}'.format(epoch+1, gen_loss[1], disc_loss[-1]))
            return gen_loss, disc_loss

            
if __name__ == "__main__":
    print("working")
    #create gen, check sumary using [model_name].summary()

    #create dic, check sumary using [model_name].summary()

    #pass into gan, check sumary using [model_name].summary()

    #
    g = Gen(28,28,1,.4)
    gen = g.main()
    #gen.summary()
    d = Disc(.2, .4, input_shape=(28,28,1))
    disc = d.main()
    #disc.summary()
    a = GAN(gen, disc)
    a.main()
    a.gan.summary()
    a.train()
    
    #a.train()

