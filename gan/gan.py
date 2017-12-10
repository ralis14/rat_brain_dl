from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import  Input, Flatten, UpSampling2D, Dense, Activation, BatchNormalization, Conv2DTranspose, LeakyReLU, Conv2D, Dropout, Reshape
from keras.optimizers import Adam, RMSprop
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.datasets import mnist

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
        self.alpha = .2
    def main(self):
        temp = self.dim1*self.dim2*self.start_depth
        model = Sequential()
        model.add(Dense(temp, input_dim=100))
        model.add(Dropout(self.drop))
        model.add(Activation('tanh'))
        model.add(Reshape((self.dim1, self.dim2, self.start_depth)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, 5, padding='same'))


        model.add(Dropout(self.drop))
        model.add(Activation('tanh'))
        model.add(UpSampling2D())
        model.add(Conv2D(64, 5, padding='same'))

        model.add(Activation('tanh'))
        model.add(Conv2D(1, 5, padding='same'))
        model.add(Activation('tanh'))

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
        model.add(Conv2D(64, 5, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(Activation('tanh'))
        
        model.add(Conv2D(128, 5, strides=2, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(Activation('tanh'))
        

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimize, metrics=['accuracy'])

        return model

class GAN(object):
    def __init__(self, gen, disc, batch_size=64):
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
        (xtrain, ytrain),(xtest, ytest) = mnist.load_data()
        xtrain = xtrain.astype(float)
        xtest = xtest.astype(float)
        xtest -=127.5
        xtest /=127.5
        xtrain -=127.5
        xtrain /= 127.5
        xtrain = xtrain.reshape((xtrain.shape[0],784))
        xtrain = shuffle(xtrain)
        data = xtrain[:num_samples]
        data = np.reshape(data,(data.shape[0], 28, 28, 1))
        data_test = np.reshape(xtest, (xtest.shape[0], 28, 28, 1))
        self.xtest = data_test
        self.xtrain = data
        return data
    def s_data_and_gen(self, noise_dim=100, num_samples=1000):
        x_temp = self.sample_data(num_samples=num_samples)
        x_temp_noise = np.random.uniform(-1, 1, size=[num_samples, noise_dim])
        noise_pred = self.G.predict(x_temp_noise)
        #print('x:{}, noise_pred:{}'.format(x_temp.shape, noise_pred.shape))
        data = np.concatenate((x_temp, noise_pred))
        labels = np.zeros((2*num_samples,1))
        labels[:num_samples] = 1
        labels[num_samples] = 1

        data, labels = shuffle(data,labels)

        assert data.shape[0] == labels.shape[0], 'data shape: {}, labels shape: {}'.format(data.shape, labels.shape)
        return data, labels

    """
    pretrain is a method to pretrain your discriminator,
    using real and fake data, generated images from generator and correctly labled data
    """
    def pre_train(self, model, num_samples=20000, b_size=128):
        assert num_samples <= 60000, "MNIST max sample num is 60K u passed: {}".format(num_samples)
        self.train_mode(model, mode=True)
        data, labels = self.s_data_and_gen(num_samples=num_samples)
        model.fit( data, labels, epochs=1, batch_size=b_size)

    """
    sample noise just takes in the gen input dimension or default of 100, and the number of actual samples of data you are using
    this will craete data and labels that will be used to be ran through the generator.
    """
    def sample_noise(self, noise_dim=100, num_samples=100):
        data = np.random.uniform(-1, 1, size=[num_samples, noise_dim])
        labels = np.zeros((num_samples, 1))
        labels[:] = 1 
        return data, labels

    def imgs_plot(self,epoch, data_noise, n_ex=16, dim=(4,4), figsize=(10,10)):
        generated_img = self.G.predict(data_noise)
        plt.figure(figsize=(8,8))
        #print('size is: {}'.format(generated_img.shape[0]//20))
        for i in range(generated_img.shape[0]//40):
            plt.subplot(8, 8, i+1)
            img = generated_img[i,:,:,0]
            img += 127.5
            img *=127.5
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        file_name = 'img/imgs_batch_{}.jpg'.format(epoch+1)
        plt.savefig(file_name)

    """
    Alternate the training of the Discriminator and the gan with the frozen generator weights.
    This way both are trained but sequentially learning from each other
    """
    def train(self, epochs=400, samples=2000, gen_input_dimension=100, out_verbose=True, progress_freq=5):
        disc_loss=[]
        gen_loss=[]
        loop = range(epochs)
        if out_verbose:
            loop = tqdm(loop)

        self.pre_train(self.D, num_samples=samples)
        for epoch in loop:
            #train disc
            data, labels = self.s_data_and_gen(num_samples=samples)
            self.train_mode(self.D, mode=True)
            disc_loss.append(self.D.train_on_batch(data, labels))

            #train gan
            data2, labels2 = self.sample_noise(num_samples=samples)
            self.train_mode(self.D, mode=False)
            gen_loss.append(self.gan.train_on_batch(data2, labels2))
            noise, bla = self.sample_noise(num_samples=samples)
            #imgs = self.gan.predict(noise)
            self.imgs_plot(epoch, noise)


            if out_verbose and (epoch+1) % progress_freq == 0:
                print('Epoch number {}: Gen loss: {}, Disc loss: {}'.format(epoch+1, gen_loss[-1], disc_loss[-1]))
                #self.imgs_plot(epoch, data)
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
    
    a.train()
    
    #a.train()

