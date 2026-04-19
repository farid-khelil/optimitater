import numpy as np
import tensorflow as tf
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from MLP import create_mlp_model
from CNN import create_cnn_model
from RNN import create_rnn_model
from DNN import create_dnn_model
from LSTM import create_lstm_model


def _is_automl_mode(test):
    return test in ('AUTOML', 'AUTO', 'ALL', 'UNIFIED')


def _decode_automl(individual):
    return {
        'model_type': int(individual[0]),
        'learning_rate': float(individual[1]),
        'batch_size': int(individual[2]),
        'epochs': int(individual[3]),
        'neurons': int(individual[4]),
        'filters': int(individual[5]),
        'kernel_size': int(individual[6]),
        'units': int(individual[7]),
    }


def _build_automl_model(self, cfg):
    """Build model dynamically from unified chromosome config.
    model_type: 0=MLP, 1=CNN, 2=LSTM, 3=RNN, 4=DNN
    """
    model_type = cfg['model_type']
    lr = cfg['learning_rate']

    # 0 -> MLP
    if model_type == 0:
        neurons = max(16, int(cfg['neurons']))
        dense_units = [neurons, max(16, neurons // 2)]
        model = create_mlp_model(
            self,
            n_dense_layers=2,
            dense_units=dense_units,
            dropout_rate=0.2,
            learning_rate=lr,
            optimizer_idx=0,
            activation='relu',
        )
        x_train, x_val = self.X_train, self.X_val

    # 1 -> CNN
    elif model_type == 1:
        filters = max(16, int(cfg['filters']))
        kernel = int(cfg['kernel_size'])
        model = create_cnn_model(
            self,
            n_conv_layers=2,
            conv_filters=[filters, filters, filters],
            kernel_sizes=[kernel, kernel, kernel],
            pool_sizes=[2, 2, 2],
            n_dense_layers=1,
            dense_units=[max(32, filters)],
            dropout_rate=0.2,
            learning_rate=lr,
            optimizer_idx=0,
            activation='relu',
        )
        x_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        x_val = self.X_val.reshape(self.X_val.shape[0], self.X_val.shape[1], 1)

    # 2 -> LSTM
    elif model_type == 2:
        units = max(16, int(cfg['units']))
        model = create_lstm_model(
            self,
            n_lstm_layers=1,
            lstm_units=[units, units, units],
            dropout_rate=0.2,
            rec_dropout_rate=0.1,
            n_dense_layers=1,
            dense_units=[max(32, units // 2)],
            learning_rate=lr,
            optimizer_idx=0,
            activation='tanh',
        )
        x_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        x_val = self.X_val.reshape(self.X_val.shape[0], 1, self.X_val.shape[1])

    # 3 -> RNN
    elif model_type == 3:
        units = max(16, int(cfg['units']))
        model = create_rnn_model(
            self,
            n_layers=1,
            rnn_units=units,
            n_dense=1,
            dens=[max(32, units // 2)],
            optimizer_idx=0,
            activation='tanh',
            dropout_rate=0.2,
            learning_rate=lr,
        )
        x_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        x_val = self.X_val.reshape(self.X_val.shape[0], 1, self.X_val.shape[1])

    # 4 -> DNN
    else:
        neurons = max(16, int(cfg['neurons']))
        hidden_units = [neurons, max(16, neurons // 2)]
        model = create_dnn_model(
            self,
            n_hidden_layers=2,
            hidden_units=hidden_units,
            dropout_rate=0.2,
            learning_rate=lr,
            optimizer_idx=0,
            activation='relu',
        )
        x_train, x_val = self.X_train, self.X_val

    return model, x_train, x_val


def _model_name_from_type(model_type):
    return {0: 'MLP', 1: 'CNN', 2: 'LSTM', 3: 'RNN', 4: 'DNN'}.get(int(model_type), 'UNKNOWN')

def evaluate_best_model(self, test='MLP'):
        """Évaluation complète du meilleur modèle sur le test set"""
        print(f"🎯 Évaluation du meilleur modèle {test}...")

        if _is_automl_mode(test):
            cfg = _decode_automl(self.best_individual)
            model_name = _model_name_from_type(cfg['model_type'])
            print(f"🤖 AutoML best model selected: {model_name}")

            best_model, x_train, x_val = _build_automl_model(self, cfg)

            if cfg['model_type'] == 1:
                x_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
            elif cfg['model_type'] in (2, 3):
                x_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])
            else:
                x_test = self.X_test

            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )

            epochs = cfg['epochs'] if cfg['model_type'] != 0 else getattr(self, 'mlp_epochs', 100)

            history = best_model.fit(
                x_train, self.y_train,
                validation_data=(x_val, self.y_val),
                epochs=epochs,
                batch_size=cfg['batch_size'],
                callbacks=[early_stopping],
                verbose=1
            )

            y_pred = best_model.predict(x_test, verbose=0, batch_size=1024)

            if self.n_classes == 2:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)

            self.best_metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred_classes),
                'precision': precision_score(self.y_test, y_pred_classes, average='weighted'),
                'recall': recall_score(self.y_test, y_pred_classes, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred_classes, average='weighted')
            }

            return best_model
        
        # Création du meilleur modèle
        #best_model = create_mlp_model(self, self.best_individual[0],self.best_individual[1:6], self.best_individual[6] ,self.best_individual[7],self.best_individual[8] ,self.best_individual[9])
        if test == 'RNN':
            best_model = create_rnn_model(self, n_layers=self.best_individual[0], rnn_units=self.best_individual[1], n_dense=self.best_individual[2], dens=self.best_individual[3:8], optimizer_idx=self.best_individual[8], activation=self.best_individual[9], dropout_rate=self.best_individual[10], learning_rate=self.best_individual[11])
            batch_size  = self.best_individual[12]
            epochs = self.best_individual[13]
        elif test == 'MLP':
            best_model = create_mlp_model(self, n_dense_layers=self.best_individual[0], dense_units=self.best_individual[1:6], dropout_rate=self.best_individual[6], learning_rate=self.best_individual[7], optimizer_idx=self.best_individual[8], activation=self.best_individual[9])
            batch_size  = self.best_individual[10]
            epochs = getattr(self, 'mlp_epochs', 100)
        elif test == 'CNN':
            best_model = create_cnn_model(self, n_conv_layers=self.best_individual[0], conv_filters=self.best_individual[1:4], kernel_sizes=self.best_individual[4:7], pool_sizes=self.best_individual[7:10], n_dense_layers=self.best_individual[10], dense_units=self.best_individual[11:16], dropout_rate=self.best_individual[16], learning_rate=self.best_individual[17], optimizer_idx=self.best_individual[18], activation=self.best_individual[19])
            batch_size  = self.best_individual[20]
            epochs = self.best_individual[21]   
        elif test == 'DNN':
            best_model = create_dnn_model(self, n_hidden_layers=self.best_individual[0], hidden_units=self.best_individual[1:6], dropout_rate=self.best_individual[6], learning_rate=self.best_individual[7], optimizer_idx=self.best_individual[8], activation=self.best_individual[9])
            batch_size = self.best_individual[10]
            epochs = self.best_individual[11]
        elif test == 'LSTM':
            best_model = create_lstm_model(self, n_lstm_layers=self.best_individual[0], lstm_units=self.best_individual[1:4], dropout_rate=self.best_individual[4], rec_dropout_rate=self.best_individual[5], n_dense_layers=self.best_individual[6], dense_units=self.best_individual[7:10], learning_rate=self.best_individual[10], optimizer_idx=self.best_individual[11], activation=self.best_individual[12])
            batch_size = self.best_individual[13]
            epochs = self.best_individual[14]
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        # Entraînement complet
        history = best_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Prédictions sur test set
        y_pred = best_model.predict(self.X_test, verbose=0, batch_size=1024)
        
        if self.n_classes == 2:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calcul des métriques
        self.best_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred_classes),
            'precision': precision_score(self.y_test, y_pred_classes, average='weighted'),
            'recall': recall_score(self.y_test, y_pred_classes, average='weighted'),
            'f1_score': f1_score(self.y_test, y_pred_classes, average='weighted')
        }
        
        return best_model

def evaluate_individual(self, individual,test='MLP'):
        """Évaluation d'un individu avec gestion mémoire"""
        try:
            if _is_automl_mode(test):
                cfg = _decode_automl(individual)
                model, x_train, x_val = _build_automl_model(self, cfg)

                early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0
                )

                epochs = cfg['epochs'] if cfg['model_type'] != 0 else getattr(self, 'mlp_epochs', 100)

                history = model.fit(
                    x_train, self.y_train,
                    validation_data=(x_val, self.y_val),
                    epochs=epochs,
                    batch_size=cfg['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Fitness objective for AutoML: validation F1-score (weighted)
                y_pred = model.predict(x_val, verbose=0, batch_size=1024)
                if self.n_classes == 2:
                    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                else:
                    y_pred_classes = np.argmax(y_pred, axis=1)
                fitness = float(f1_score(self.y_val, y_pred_classes, average='weighted'))

                # Keep best tested individual per model type (unique models in final ranking)
                if not hasattr(self, '_automl_best_by_model') or self._automl_best_by_model is None:
                    self._automl_best_by_model = {}

                mtype = int(cfg['model_type'])
                best_entry = self._automl_best_by_model.get(mtype)
                if best_entry is None or fitness > best_entry['fitness']:
                    self._automl_best_by_model[mtype] = {
                        'individual': list(individual),
                        'fitness': float(fitness),
                    }

                del model, history, y_pred, y_pred_classes
                tf.keras.backend.clear_session()
                gc.collect()

                return (fitness,)

            # Création du modèle
            #model = create_mlp_model(self, individual[0],individual[1:6], individual[6] ,individual[7],individual[8] ,individual[9])
            if test == 'RNN':
                model = create_rnn_model(self, n_layers=individual[0], rnn_units=individual[1], n_dense=individual[2], dens=individual[3:8], optimizer_idx=individual[8], activation=individual[9], dropout_rate=individual[10], learning_rate=individual[11])
                batch_size  = individual[12]
                epochs = individual[13]
            elif test == 'MLP':
                model = create_mlp_model(self, individual[0], individual[1:6], individual[6], individual[7], individual[8], individual[9])
                batch_size  = individual[10]
                epochs = getattr(self, 'mlp_epochs', 100)
            elif test == 'CNN':
                model = create_cnn_model(self, n_conv_layers=individual[0], conv_filters=individual[1:4], kernel_sizes=individual[4:7], pool_sizes=individual[7:10], n_dense_layers=individual[10], dense_units=individual[11:16], dropout_rate=individual[16], learning_rate=individual[17], optimizer_idx=individual[18], activation=individual[19])
                batch_size  = individual[20]
                epochs = individual[21]
            elif test == 'DNN':
                model = create_dnn_model(self, n_hidden_layers=individual[0], hidden_units=individual[1:6], dropout_rate=individual[6], learning_rate=individual[7], optimizer_idx=individual[8], activation=individual[9])
                batch_size = individual[10]
                epochs = individual[11]
            elif test == 'LSTM':
                model = create_lstm_model(self, n_lstm_layers=individual[0], lstm_units=individual[1:4], dropout_rate=individual[4], rec_dropout_rate=individual[5], n_dense_layers=individual[6], dense_units=individual[7:10], learning_rate=individual[10], optimizer_idx=individual[11], activation=individual[12])
                batch_size = individual[13]
                epochs = individual[14]
                
            
            # Déterminer la taille de batch
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=0
            )
            
            # Entraînement
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Prédictions sur validation
            y_pred = model.predict(self.X_val, verbose=0, batch_size=1024)
            
            if self.n_classes == 2:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)
            
            # CALCUL DU RECALL (METRIQUE PRINCIPALE)
            recall = recall_score(self.y_val, y_pred_classes, average='weighted')
            
            # Nettoyage mémoire
            del model, history, y_pred, y_pred_classes
            tf.keras.backend.clear_session()
            gc.collect()
            
            return (recall,)
            
        except Exception as e:
            print(individual)
            print(f"⚠️ Erreur lors de l'évaluation: {str(e)}")
            return (0.0,)