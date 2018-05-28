import keras
from keras import backend as K, Input, Model
from keras.datasets import mnist
from keras.layers import Dense
from keras.regularizers import l2


def get_dataset():
    num_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.reshape([-1, 28**2])
    x_test = x_test.reshape([-1, 28**2])

    return x_train, y_train, (x_test, y_test), input_shape


def get_mlp(num_layers=3, lantent_size=100, l2_reg=0.0001, activation='relu'):
    in_vec = Input(shape=[784])

    x = in_vec
    for i in range(num_layers):
        x = Dense(
            lantent_size,
            activation=activation,
            kernel_regularizer=l2(l2_reg)
        )(x)

    y = Dense(10, activation='softmax')(x)
    model = Model(in_vec, y)
    return model