
from random import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

def get_mlp_param():
    param = {
        'model__n_dense_layers': [1, 2, 3, 4, 5],
        'model__dense_units': [[64, 128, 256, 512], [32, 64, 128, 256], [16, 32, 64, 128], [128, 256, 512, 1024], [256, 512, 1024, 2048], [64, 128, 256], [32, 64, 128], [16, 32, 64], [128, 256, 512], [256, 512, 1024]],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
        'model__learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.05],
        'model__optimizer_idx': [0, 1, 2],
        'model__activation_idx': [0, 1, 2, 3],
        'batch_size': [16, 32, 64, 128],
        }

    return param

def create_mlp_model(obj, n_dense_layers=2, dense_units=[52,64], dropout_rate=0.2, learning_rate=0.01, optimizer_idx=0, activation_idx=0):
        """
        Création du modèle MLP avec gestion dynamique des couches
        """
        # Décodage de l'individu
        # n_dense_layers = individual[0]  # 1-5
        # dense_units = individual[1:6]  # Unités pour chaque couche
        # dropout_rate = individual[6]    # Taux de dropout
        # learning_rate = individual[7]   # Taux d'apprentissage
        # optimizer_idx = individual[8]   # Index de l'optimiseur
        # activation_idx = individual[9]  # Index de la fonction d'activation
        # batch_size_idx = individual[10] # Index du batch size
        
        # Mappage des index aux valeurs réelles
        activations = ['relu', 'elu', 'selu', 'tanh']
        activation = activations[activation_idx]
        
        model = Sequential()
        
        # Première couche avec input_shape
        model.add(Dense(
            units=dense_units[0],
            activation=activation,
            input_shape=(obj.n_features,)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Couches cachées supplémentaires
        for i in range(1, n_dense_layers):
            model.add(Dense(
                units=dense_units[i],
                activation=activation
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Couche de sortie
        if obj.n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(obj.n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Optimiseur
        optimizers = [
            Adam(learning_rate=learning_rate),
            RMSprop(learning_rate=learning_rate),
            SGD(learning_rate=learning_rate, momentum=0.9)
        ]
        
        model.compile(
            optimizer=optimizers[optimizer_idx],
            loss=loss,
            metrics=['accuracy']
        )
        
        return model