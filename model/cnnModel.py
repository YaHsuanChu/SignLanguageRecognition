from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, RandomRotation, RandomContrast
from keras.optimizers import adam_v2
import dataset

(X_train, y_train), (X_test, y_test) = dataset.readDataResize(".\data\ours+American Sign Language Letters.v1-v1.multiclass+asl_alphabet(700)_train", 128, 128),  dataset.readDataResize("data\ours+American Sign Language Letters.v1-v1.multiclass+asl_alphabet(700)_test", 128, 128)

X_train = X_train.reshape(-1, 128, 128, 3)/255.
X_test = X_test.reshape(-1, 128, 128, 3)/255.

import gc
gc.collect()

model = Sequential()

model.add(RandomRotation(0.2))
model.add(RandomContrast(0.2))

model.add(Convolution2D(
    batch_input_shape=(None, 128, 128, 3),
    filters=32,
    kernel_size=3,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_last',
))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_last',
))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(128, 3, strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(256, 3, strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Fully connected layer 2 to shape (29) for 29 classes
model.add(Dense(29))
model.add(Activation('softmax'))

model.build((None, 128, 128, 3))
print(model.summary())

adam = adam_v2.Adam(learning_rate = 1e-3, decay = 1e-5)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')

model.fit(X_train, y_train, epochs=50, batch_size=8, shuffle=True)

model.save('cnn_model.h5')

gc.collect()

print('\nTesting ------------')

# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model = None
