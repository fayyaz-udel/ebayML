from keras import Input
from keras.layers import Dense, BatchNormalization, Concatenate, ReLU, Dropout
from keras.models import Model


def build_model():
    inputs = Input(shape=(27,))

    l1 = Dense(512)(inputs)
    l1 = BatchNormalization()(l1)
    l1 = ReLU()(l1)
    l1 = Dropout(0.2)(l1)

    l2 = Dense(512)(l1)
    l2 = BatchNormalization()(l2)
    l2 = ReLU()(l2)
    l2 = Dropout(0.2)(l2)

    l3 = Dense(128)(l2)
    l3 = BatchNormalization()(l3)
    l3 = ReLU()(l3)

    l4 = Concatenate()([l3, inputs])
    #l4 = Dense(128)(l4)
    #l4 = BatchNormalization()(l4)
    #l4 = ReLU()(l4)
    l4 = Dense(64)(l4) # l4 = Dense(32)(l4)
    l4 = BatchNormalization()(l4)
    l4 = ReLU()(l4)

    outputs = Dense(1, activation='linear')(l4)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model
