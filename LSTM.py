

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization



def get_lstm_param():
    param = {
        'model__n_lstm_layers': [1, 2, 3],
        'model__lstm_units': [[32, 64], [64, 128], [128, 256], [32, 64, 128], [64, 128, 256]],
        'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
        'model__rec_dropout_rate': [0.0, 0.1, 0.2],
        'model__n_dense_layers': [1, 2],
        'model__dense_units': [[64], [128], [64, 128], [128, 256]],
        'model__learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.05],
        'model__optimizer_idx': [0, 1, 2],
        'model__activation_idx': [0, 1, 2, 3],
        'batch_size': [16, 32, 64],
        }

    return param

def create_lstm_model(obj, n_lstm_layers=2, lstm_units=[64, 128], dropout_rate=0.2, rec_dropout_rate=0.2, n_dense_layers=1, dense_units=[64], learning_rate=0.001, optimizer_idx=0, activation='relu'):
        """
        Création du modèle LSTM avec gestion dynamique des couches
        """
        # Décodage de l'individu
        # n_lstm_layers = individual[0]    # 1-3
        # lstm_units = individual[1:4]      # Unités pour chaque couche LSTM
        # dropout_rate = individual[4]       # Taux de dropout
        # rec_dropout_rate = individual[5]   # Taux de dropout récurrent
        # n_dense_layers = individual[6]    # 1-2
        # dense_units = individual[7:9]     # Neurones par couche dense
        # learning_rate = individual[9]      # Taux d'apprentissage
        # optimizer_idx = individual[10]     # Index de l'optimiseur
        # activation_idx = individual[11]    # Index de la fonction d'activation
        # batch_size_idx = individual[12]    # Index du batch size
        
        # Mappage des index aux valeurs réelles
        model = Sequential()
        sequence_length = max(1, int(getattr(obj, 'sequence_length', 1)))
        features_per_timestep = int(getattr(obj, 'features_per_timestep', obj.n_features))
        
        # Couches LSTM (seulement le nombre utilisé)
        for i in range(n_lstm_layers):
            # Configuration de la première couche
            if i == 0:
                return_sequences = (n_lstm_layers > 1)
                model.add(LSTM(
                    units=lstm_units[i],
                    input_shape=(sequence_length, features_per_timestep),
                    dropout=dropout_rate,
                    recurrent_dropout=rec_dropout_rate,
                    return_sequences=return_sequences
                ))
            # Couches intermédiaires
            elif i < n_lstm_layers - 1:
                model.add(LSTM(
                    units=lstm_units[i],
                    dropout=dropout_rate,
                    recurrent_dropout=rec_dropout_rate,
                    return_sequences=True
                ))
            # Dernière couche LSTM
            else:
                model.add(LSTM(
                    units=lstm_units[i],
                    dropout=dropout_rate,
                    recurrent_dropout=rec_dropout_rate
                ))
            
            model.add(BatchNormalization())
        
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
