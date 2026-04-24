

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

def get_cnn_param():
    param = {
        'model__n_conv_layers': [1, 2, 3],
        'model__conv_filters': [[32, 64, 128], [16, 32, 64], [64, 128, 256]],
        'model__kernel_sizes': [[3, 3, 3], [5, 5, 5], [3, 5, 7]],
        'model__pool_sizes': [[2, 2, 2], [4, 4, 4], [2, 4, 8]],
        'model__n_dense_layers': [1, 2, 3, 4, 5],
        'model__dense_units': [[64, 128, 256, 512], [32, 64, 128, 256], [16, 32, 64, 128], [128, 256, 512, 1024], [256, 512, 1024, 2048], [64, 128, 256], [32, 64, 128], [16, 32, 64], [128, 256, 512], [256, 512, 1024]],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
        'model__learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.05],
        'model__optimizer_idx': [0, 1, 2],
        'model__activation_idx': [0, 1, 2, 3],
        'batch_size': [16, 32, 64],
        }

    return param

def create_cnn_model(obj, n_conv_layers=2, conv_filters=[32, 64], kernel_sizes=[3, 3], pool_sizes=[2, 2], n_dense_layers=1, dense_units=[64], dropout_rate=0.2, learning_rate=0.001, optimizer_idx=0, activation='relu'):
        """
        Création du modèle CNN avec gestion dynamique des couches
        et nouveaux hyperparamètres
        """
        # Décodage de l'individu
        # n_conv_layers = individual[0]  # 1-3
        # conv_filters = individual[1:4]  # Filtres pour chaque couche
        # kernel_sizes = individual[4:7]  # Tailles de kernel
        # pool_sizes = individual[7:10]   # Tailles de pooling (nouveau)
        # n_dense_layers = individual[10] # 1-2
        # dense_units = individual[11:13] # Neurones par couche dense
        # dropout_rate = individual[13]    # Taux de dropout
        # learning_rate = individual[14]   # Taux d'apprentissage
        # optimizer_idx = individual[15]   # Index de l'optimiseur
        # activation_idx = individual[16]  # Index de la fonction d'activation (nouveau)
        # batch_size_idx = individual[17]  # Index du batch size (nouveau)# Mappage des index aux valeurs réelles
        
        model = Sequential()
        
        # Couches convolutionnelles (seulement le nombre utilisé)
        for i in range(n_conv_layers):
            # Première couche avec input_shape
            if i == 0:
                model.add(Conv1D(
                    filters=conv_filters[i],
                    kernel_size=kernel_sizes[i],
                    activation=activation,
                    input_shape=(obj.n_features, 1),
                    padding='same'
                ))
            # Couches suivantes
            else:
                model.add(Conv1D(
                    filters=conv_filters[i],
                    kernel_size=kernel_sizes[i],
                    activation=activation,
                    padding='same'
                ))
            
            model.add(BatchNormalization())
            
            # Pooling seulement si la dimension le permet
            if model.output_shape[1] > pool_sizes[i]:
                model.add(MaxPooling1D(pool_size=pool_sizes[i]))
        
        model.add(Flatten())
        model.add(Dropout(dropout_rate))
        
        # Couches denses (seulement le nombre utilisé)
        for i in range(n_dense_layers):
            model.add(Dense(dense_units[i], activation=activation))
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