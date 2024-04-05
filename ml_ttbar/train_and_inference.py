import os 
import numpy as np
from utils import *
from CONSTANTS import *


def generate_training_jet_data():
    import keras

    outdir = 'images_out/'
    assert os.path.isdir(outdir)

    data0 = np.load(outdir + 'qcd_leading_jet.npz', allow_pickle=True)['arr_0']
    data1 = np.load(outdir + 'tt_leading_jet.npz', allow_pickle=True)['arr_0']

    print('We have %d QCD jets and %d top jets' % (len(data0), len(data1)))

    data0_len = len(data0)

    x_data = np.concatenate((data0, data1))
    del data0
    del data1
    x_data = np.asarray(x_data, dtype=float)
    
    # pad and normalize images
    x_data = list(map(pad_image, x_data))
    x_data = list(map(normalize, x_data))

    all_data = [[image, int(i >= data0_len)] for i, image in enumerate(x_data)]
    del x_data
    all_data = np.asarray(all_data, dtype=np.object_)

    np.random.seed(0)  # for reproducibility
    x_data, y_data = np.random.permutation(all_data).T

    del all_data

    # the data coming out of previous commands is a list of 2D arrays. We want a 3D np array (n_events, xpixels, ypixels)
    x_data = np.stack(x_data)

    print(x_data.shape, y_data.shape)

    # reshape for tensorflow: x_data.shape + (1,) = shortcut for (x_data.shape[0], MAX_HEIGHT, MAX_WIDTH, 1)
    x_data = x_data.reshape(x_data.shape + (1,)).astype('float32')

    y_data = keras.utils.to_categorical(y_data, 2)

    print(x_data.shape, y_data.shape)
    
    n_train = math.floor(len(x_data) * 0.8) 
    (x_train, x_test) = x_data[:n_train], x_data[n_train:]
    (y_train, y_test) = y_data[:n_train], y_data[n_train:]

    print('We will train+validate on %d images, leaving %d for cross-validation' % (n_train, len(x_data) - n_train))

    # ---------------------------------

    model_dir = 'trained_models_low_pt/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # ---------------------------------

    np.savez(model_dir + 'x_train.npz', x_train)
    np.savez(model_dir + 'y_train.npz', y_train)

    np.savez(model_dir + 'x_test.npz', x_test)
    np.savez(model_dir + 'y_test.npz', y_test)




def train_jets():
    model_dir = 'trained_models_low_pt/' 
    assert os.path.isdir(model_dir)

    # ---------------------------------

    x_train = np.load(model_dir + 'x_train.npz')['arr_0']
    y_train = np.load(model_dir + 'y_train.npz')['arr_0']

    # ---------------------------------

    from keras.models import Sequential
    # from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
    from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

    # ---------------------------------

    print("Model 0")

    model0 = Sequential()
    model0.add(Flatten(input_shape=(MAX_HEIGHT, MAX_WIDTH, 1)))  # Images are a 3D matrix, we have to flatten them to be 1D
    model0.add(Dense(2, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model0.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history_logi = model0.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=100, shuffle=True, verbose=0)

    # ---------------------------------

    print("Model 0 Save")

    model0.save(model_dir + 'logi.h5')

    # ---------------------------------

    print("Model 1")

    model1 = Sequential()
    model1.add(Flatten(input_shape=(MAX_HEIGHT, MAX_WIDTH, 1)))  # Images are a 3D matrix, we have to flatten them to be 1D
    model1.add(Dense(100, kernel_initializer='normal', activation='tanh'))
    model1.add(Dropout(0.5))  # drop a unit with  50% probability.

    model1.add(Dense(100, kernel_initializer='orthogonal', activation='tanh'))
    model1.add(Dropout(0.5))  # drop a unit with  50% probability.

    model1.add(Dense(100, kernel_initializer='orthogonal', activation='tanh'))
    # model.add(Activation('sigmoid'))
    model1.add(Dense(2, kernel_initializer='normal', activation='softmax'))  # last layer, this has a softmax to do the classification

    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history_mlp = model1.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=100, shuffle=True, verbose=1)

    # ---------------------------------

    print("Model 1 Save")

    model1.save(model_dir + 'mlp.h5')

    # ---------------------------------

    print("Model CNN")

    model_cnn = Sequential()
    model_cnn.add(Conv2D(32, (3, 3), input_shape=(MAX_HEIGHT, MAX_WIDTH, 1), activation='relu'))
    model_cnn.add(Conv2D(32, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))

    model_cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_cnn.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))

    model_cnn.add(Flatten())
    model_cnn.add(Dense(300, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(2, activation='softmax'))

    # Compile model
    model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_cnn = model_cnn.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=100, shuffle=True, verbose=1)

    # ---------------------------------

    print("Model CNN Save")

    model_cnn.save(model_dir + 'cnn.h5')

    # ---------------------------------

    print("Training History Save")

    np.savez(model_dir + 'training_histories.npz', [history.history for history in [history_logi, history_mlp, history_cnn]])

def predict_jets():
    model_dir = 'trained_models_low_pt/'  
    assert os.path.isdir(model_dir)

    # ---------------------------------

    import keras

    model0 = keras.models.load_model(model_dir + 'logi.h5')
    model1 = keras.models.load_model(model_dir + 'mlp.h5')
    model_cnn = keras.models.load_model(model_dir + 'cnn.h5')

    x_test = np.load(model_dir + 'x_test.npz')['arr_0']

    predictions0 = model0.predict(x_test)
    predictions1 = model1.predict(x_test)
    predictions_cnn = model_cnn.predict(x_test)

    # ---------------------------------

    np.savez(model_dir + 'predictions0.npz', predictions0)
    np.savez(model_dir + 'predictions1.npz', predictions1)
    np.savez(model_dir + 'predictions_cnn.npz', predictions_cnn)
