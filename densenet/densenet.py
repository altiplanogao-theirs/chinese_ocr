from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPool2D
from keras.layers import Input, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter

def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if(dropout_rate):
        x = Dropout(dropout_rate)(x)

    if(pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif(pooltype == 1):
        x = ZeroPadding2D(padding = (0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif(pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter


def _normal_conv_block(input, layers, kernel_size, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Conv2D(layers, kernel_size, kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x


def _normal_pooling(input):
    x = MaxPool2D((2, 2), strides=(2, 2))(input)
    return x


def normal_cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    x = _normal_conv_block(input, 32, (3, 3))
    x = _normal_conv_block(x, 32, (3, 3), dropout_rate=_dropout_rate)
    x = _normal_pooling(x)

    x = _normal_conv_block(x, 64, (3, 3))
    x = _normal_conv_block(x, 64, (3, 3), dropout_rate=_dropout_rate)
    x = _normal_pooling(x)

    x = _normal_conv_block(x, 128, (3, 3))
    x = _normal_conv_block(x, 128, (3, 3), dropout_rate=_dropout_rate)
    x = _normal_pooling(x)

    x2 = Conv2D(512, (4, 4), kernel_initializer='he_normal', padding='same', strides=(4, 1),
                use_bias=False, kernel_regularizer=l2(_weight_decay))(x)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(512, (1, 1), kernel_initializer='he_normal', padding='valid',
                use_bias=False, kernel_regularizer=l2(_weight_decay))(x2)
    x3 = Activation('relu')(x3)
    x4 = Permute((2, 1, 3), name='permute')(x3)
    x4 = TimeDistributed(Flatten(), name='flatten')(x4)
    y_pred = Dense(nclass, name='out', activation='softmax')(x4)

    return y_pred

    # x = Conv2D(_nb_filter, (5, 5), kernel_initializer='he_normal', padding='same',
    #            use_bias=False, kernel_regularizer=l2(_weight_decay))(input)
    # x = Conv2D(_nb_filter, (5, 5), kernel_initializer='he_normal', padding='same',
    #            use_bias=False, kernel_regularizer=l2(_weight_decay))(x)
    #
    # # 64 + 8 * 8 = 128
    # x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # # 128
    # x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)
    #
    # # 128 + 8 * 8 = 192
    # x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # # 192 -> 128
    # x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)
    #
    # # 128 + 8 * 8 = 192
    # x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    #
    # x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    # x = Activation('relu')(x)
    #
    # x = Permute((2, 1, 3), name='permute')(x)
    # x = TimeDistributed(Flatten(), name='flatten')(x)
    # y_pred = Dense(nclass, name='out', activation='softmax')(x)
    #
    # # basemodel = Model(inputs=input, outputs=y_pred)
    # # basemodel.summary()
    #
    # return y_pred


def dense_cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()

    return y_pred

def dense_blstm(input):

    pass

# input = Input(shape=(32, 280, 1), name='the_input')
# yy = dense_cnn(input, 5000)
#
# basemodel = Model(inputs=input, outputs=yy)
# basemodel.summary()
#
# yy2 = normal_cnn(input, 5000)
#
# basemodel2 = Model(inputs=input, outputs=yy2)
# basemodel2.summary()
