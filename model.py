import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, Softmax, LeakyReLU, ReLU, Input, BatchNormalization, Reshape, Flatten
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import SquaredHinge 
from keras.preprocessing.image import ImageDataGenerator

from image import ImageProcessor
import numpy as np
import matplotlib.pyplot as plt 

im = ImageProcessor()

IMG_ROWS = 1280
IMG_COLS = 920
CHANNELS = 3 
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)
NOISE_SHAPE = (20,)

EPOCHS = 50 
BATCH_SIZE = 2 
SAVE_INTERVAL = 5

def save_imgs(epoch):
    r,c = 5,5 
    noise = np.random.normal(0,1,(r*c,100))
    gen_imgs = generator.predict(noise)
    
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplot(r,c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0])
            axs[i,j].axis('off')
            cnt+=1
    fig.savefig("C:/Users/Nikola Kojadinovic/digital-art-gan/training_out")
    plt.close
    
    
def build_generator():
    
    model = Sequential()
    
    model.add(Dense(32, input_shape = NOISE_SHAPE))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(IMG_SHAPE), activation = 'tanh'))
    model.add(Reshape(IMG_SHAPE))
    
    model.summary()
    
    noise = Input(shape = NOISE_SHAPE)
    img = model(noise)    
    
    return Model(noise,img)
    
    
        
def build_discriminator():
    
    model = Sequential()
    
    model.add(Flatten(input_shape = IMG_SHAPE))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    
    img = Input(shape = IMG_SHAPE)
    validity = model(img)
    
    return Model(img, validity)

def train(epochs, batch_size, save_interval=500):
    
    datagen = ImageDataGenerator()
    X_train = datagen.flow_from_directory("images_small")
    print()
    
    half_batch = BATCH_SIZE // 2
    
    for epoch in range(epochs):
    
        #training a discriminator
        idx = np.random.randint(0,14, half_batch)
        imgs = X_train[idx]
        noise = np.random.normal(0,1, (half_batch,20))
        gen_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch,1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch,1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        #train a generator
        noise = np.random.normal(0,1,(half_batch,20))
        valid_y = np.array([1]*batch_size)
        g_loss = combined.train_on_batch(noise,valid_y)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1],g_loss))
          
        if epoch % save_interval == 0:
            save_imgs(epoch)    
    

optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss = 'binary_crossentropy',
                      optimizer = optimizer,
                      metrics = ['accuracy'])


generator = build_generator()
generator.compile(loss = 'binary_crossentropy', optimizer = optimizer)

z = Input(shape = (20, ))
img = generator(z)

discriminator.trainable = False 
valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss = 'binary_crossentropy', optimizer=optimizer)

train(epochs = EPOCHS, 
      batch_size=BATCH_SIZE, 
      save_interval=SAVE_INTERVAL)
 
generator.save('generator_model_test.h5')
print('Ok')