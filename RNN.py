

from pyexpat import model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

def create_rnn_model(obj,n_layers=1,rnn_units=64,n_dense=2,dens=[64,64],optimizer_idx=0, activation='relu', dropout_rate=0.2,learning_rate=0.001):

    model = Sequential()

    model.add(SimpleRNN(
        units=rnn_units,
        input_shape=(1, obj.n_features),
        dropout=dropout_rate,
        return_sequences=True

    ))

    for i in range(n_layers - 1):
        model.add(SimpleRNN(
            units=rnn_units,
            dropout=dropout_rate,
            return_sequences=True
        ))

        model.add(BatchNormalization())
    
    model.add(SimpleRNN(
        units=rnn_units,
        dropout=dropout_rate,
        return_sequences=False
    ))
    for i in range(n_dense):
        model.add(Dense(dens[i], activation=activation))
        model.add(Dropout(dropout_rate))
    
    if obj.n_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(obj.n_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    optimalizers = [
            Adam(learning_rate=learning_rate),
            RMSprop(learning_rate=learning_rate),
            SGD(learning_rate=learning_rate, momentum=0.9)
    ]
    model.compile(
        optimizer=optimalizers[optimizer_idx],
        loss=loss,
        metrics=['accuracy']
    )

    return model