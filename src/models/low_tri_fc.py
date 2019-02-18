import numpy as np
from keras.layers import Conv1D, Reshape, Flatten
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Lambda, Dropout, concatenate, BatchNormalization
from keras import optimizers, regularizers

def define_model_fc_w_conv():
    inp = Input((imsize, imsize))
    
    layer = concatenate(
        [Dense(1, activation='relu', input_shape=(i+1,))(Lambda(lambda x: x[:,i,:i+1])(inp))
         for i in np.arange(imsize)], axis=-1)
    layer = Reshape((imsize, 1), input_shape=(imsize,))(layer)
    
    layer = Dropout(0.25)((Conv1D(19, 8, activation='relu')(layer)))
    
    layer = Flatten()(layer)
    
    layer = Dropout(0.25)((Dense(128, activation='relu')(layer)))
     
    outp = Dense(2, activation='sigmoid')(layer)
    
    model = Model(inputs=inp, outputs=outp)
    
    opt = optimizers.RMSprop()
    opt = optimizers.SGD(lr=0.01,  decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def define_model_fc():
    inp = Input((imsize, imsize))
    
    layer = concatenate(
        [Dense(1, activation='relu', input_shape=(i+1,))(Lambda(lambda x: x[:,i,:i+1])(inp))
         for i in np.arange(imsize)], axis=-1)
    
    layer = Dropout(0.25)((Dense(128, activation='relu')(layer)))
    
    # layer = Dropout(0.25)((Dense(32, activation='relu')(layer)))
     
    outp = Dense(2, activation='sigmoid')(layer)
    
    model = Model(inputs=inp, outputs=outp)
    
    opt = optimizers.RMSprop()
    opt = optimizers.SGD(lr=0.01,  decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

