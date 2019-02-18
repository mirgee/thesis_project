from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras import optimizers
from keras import initializers

def nikitas_model():
    model = Sequential()
    ki = initializers.RandomNormal(0, 0.1, 23)
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(image_height,image_width,num_channels),
                     kernel_initializer=ki))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer=ki))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.75))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=ki))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=ki))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.75))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=ki))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    # model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(lr=1e-4, beta_1=0.999, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
