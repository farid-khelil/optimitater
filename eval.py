import numpy as np
import tensorflow as tf
import gc
from sklearn.model_selection import train_test_split
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
        'neurons': int(individual[4]),
        'filters': int(individual[5]),
        'kernel_size': int(individual[6]),
        'units': int(individual[7]),
        'n_conv_layers': int(individual[8]),
        'pool_size': int(individual[9]),
        'n_dense_layers': int(individual[10]),
        'dense_units': int(individual[11]),
        'dropout_rate': float(individual[12]),
        'optimizer_idx': int(individual[13]),
        'activation': individual[14],
    }


def _get_sequence_length(obj):
    """Default to single-step sequences unless explicitly configured."""
    return max(1, int(getattr(obj, 'sequence_length', 1)))


def _reshape_recurrent_input(obj, X):
    """Prepare RNN/LSTM tensors as (samples, timesteps, features_per_timestep)."""
    X = np.asarray(X)
    if X.ndim == 3:
        return X
    if X.ndim != 2:
        raise ValueError(f"Expected 2D or 3D recurrent input, got shape {X.shape}")

    seq_len = _get_sequence_length(obj)
    if seq_len == 1:
        return X.reshape(X.shape[0], 1, X.shape[1])

    step_features = getattr(obj, 'features_per_timestep', None)
    if step_features is not None:
        step_features = int(step_features)
        expected = seq_len * step_features
        if X.shape[1] != expected:
            raise ValueError(
                f"Invalid recurrent input width {X.shape[1]} for sequence_length={seq_len} "
                f"and features_per_timestep={step_features} (expected {expected})."
            )
    else:
        if X.shape[1] % seq_len != 0:
            raise ValueError(
                f"Cannot reshape input width {X.shape[1]} into sequence_length={seq_len}. "
                "Set obj.features_per_timestep explicitly."
            )
        step_features = X.shape[1] // seq_len

    return X.reshape(X.shape[0], seq_len, step_features)


def _reshape_cnn_input(X):
    X = np.asarray(X)
    if X.ndim == 3:
        return X
    if X.ndim != 2:
        raise ValueError(f"Expected 2D or 3D CNN input, got shape {X.shape}")
    return X.reshape(X.shape[0], X.shape[1], 1)


def _prepare_inputs_for_model(self, model_name):
    model_name = str(model_name).upper()
    if model_name in ('RNN', 'LSTM'):
        return (
            _reshape_recurrent_input(self, self.X_train),
            _reshape_recurrent_input(self, self.X_val),
            _reshape_recurrent_input(self, self.X_test),
        )
    if model_name == 'CNN':
        return (
            _reshape_cnn_input(self.X_train),
            _reshape_cnn_input(self.X_val),
            _reshape_cnn_input(self.X_test),
        )
    return self.X_train, self.X_val, self.X_test


def _prepare_train_val_for_model(self, model_name, X_train, X_val):
    model_name = str(model_name).upper()
    if model_name in ('RNN', 'LSTM'):
        return _reshape_recurrent_input(self, X_train), _reshape_recurrent_input(self, X_val)
    if model_name == 'CNN':
        return _reshape_cnn_input(X_train), _reshape_cnn_input(X_val)
    return X_train, X_val


def _get_training_and_validation_split(self):
    """Return a single train/validation split for fitness evaluation.

    When a full 80% training pool is available, create an internal 80/20
    train/validation split from that pool. Otherwise, fall back to the
    precomputed X_train / X_val split.
    """
    if (
        hasattr(self, 'X_train_val') and self.X_train_val is not None
        and hasattr(self, 'y_train_val') and self.y_train_val is not None
    ):
        X_source = np.asarray(self.X_train_val)
        y_source = np.asarray(self.y_train_val).astype(int)

        stratify = None
        try:
            class_counts = np.bincount(y_source)
            if class_counts.size > 0 and class_counts.min() >= 2:
                stratify = y_source
        except Exception:
            stratify = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_source,
            y_source,
            test_size=0.20,
            random_state=42,
            stratify=stratify,
        )
        return X_train, X_val, y_train, y_val

    return (
        np.asarray(self.X_train),
        np.asarray(self.X_val),
        np.asarray(self.y_train).astype(int),
        np.asarray(self.y_val).astype(int),
    )


def _build_automl_model(self, cfg, X_train=None, X_val=None):
    """Build model dynamically from unified chromosome config.
    model_type: 0=MLP, 1=CNN, 2=LSTM, 3=RNN, 4=DNN
    """
    model_type = cfg['model_type']
    lr = cfg['learning_rate']
    base_train = self.X_train if X_train is None else np.asarray(X_train)
    base_val = self.X_val if X_val is None else np.asarray(X_val)

    # 0 -> MLP
    if model_type == 0:
        neurons = max(16, int(cfg['neurons']))
        n_dense_layers = int(cfg.get('n_dense_layers', 2))
        dense_base = max(16, int(cfg.get('dense_units', neurons)))
        dense_units = [dense_base] * 5
        model = create_mlp_model(
            self,
            n_dense_layers=n_dense_layers,
            dense_units=dense_units,
            dropout_rate=float(cfg.get('dropout_rate', 0.2)),
            learning_rate=lr,
            optimizer_idx=int(cfg.get('optimizer_idx', 0)),
            activation=cfg.get('activation', 'relu'),
        )
        x_train, x_val = base_train, base_val

    # 1 -> CNN
    elif model_type == 1:
        filters = max(16, int(cfg['filters']))
        kernel = int(cfg['kernel_size'])
        n_conv_layers = int(cfg.get('n_conv_layers', 2))
        n_dense_layers = int(cfg.get('n_dense_layers', 1))
        dense_base = max(32, int(cfg.get('dense_units', filters)))
        model = create_cnn_model(
            self,
            n_conv_layers=n_conv_layers,
            conv_filters=[filters, filters, filters],
            kernel_sizes=[kernel, kernel, kernel],
            pool_sizes=[int(cfg.get('pool_size', 2))] * 3,
            n_dense_layers=n_dense_layers,
            dense_units=[dense_base] * 5,
            dropout_rate=float(cfg.get('dropout_rate', 0.2)),
            learning_rate=lr,
            optimizer_idx=int(cfg.get('optimizer_idx', 0)),
            activation=cfg.get('activation', 'relu'),
        )
        x_train = _reshape_cnn_input(base_train)
        x_val = _reshape_cnn_input(base_val)

    # 2 -> LSTM
    elif model_type == 2:
        units = max(16, int(cfg['units']))
        n_dense_layers = int(cfg.get('n_dense_layers', 1))
        dense_base = max(32, int(cfg.get('dense_units', units // 2)))
        model = create_lstm_model(
            self,
            n_lstm_layers=1,
            lstm_units=[units, units, units],
            dropout_rate=float(cfg.get('dropout_rate', 0.2)),
            rec_dropout_rate=0.1,
            n_dense_layers=n_dense_layers,
            dense_units=[dense_base] * 3,
            learning_rate=lr,
            optimizer_idx=int(cfg.get('optimizer_idx', 0)),
            activation=cfg.get('activation', 'tanh'),
        )
        x_train = _reshape_recurrent_input(self, base_train)
        x_val = _reshape_recurrent_input(self, base_val)

    # 3 -> RNN
    elif model_type == 3:
        units = max(16, int(cfg['units']))
        n_dense_layers = int(cfg.get('n_dense_layers', 1))
        dense_base = max(32, int(cfg.get('dense_units', units // 2)))
        model = create_rnn_model(
            self,
            n_layers=1,
            rnn_units=units,
            n_dense=n_dense_layers,
            dens=[dense_base] * 5,
            optimizer_idx=int(cfg.get('optimizer_idx', 0)),
            activation=cfg.get('activation', 'tanh'),
            dropout_rate=float(cfg.get('dropout_rate', 0.2)),
            learning_rate=lr,
        )
        x_train = _reshape_recurrent_input(self, base_train)
        x_val = _reshape_recurrent_input(self, base_val)

    # 4 -> DNN
    else:
        neurons = max(16, int(cfg['neurons']))
        n_hidden_layers = int(cfg.get('n_dense_layers', 2))
        dense_base = max(16, int(cfg.get('dense_units', neurons)))
        hidden_units = [dense_base] * 5
        model = create_dnn_model(
            self,
            n_hidden_layers=n_hidden_layers,
            hidden_units=hidden_units,
            dropout_rate=float(cfg.get('dropout_rate', 0.2)),
            learning_rate=lr,
            optimizer_idx=int(cfg.get('optimizer_idx', 0)),
            activation=cfg.get('activation', 'relu'),
        )
        x_train, x_val = base_train, base_val

    return model, x_train, x_val


def _model_name_from_type(model_type):
    return {0: 'MLP', 1: 'CNN', 2: 'LSTM', 3: 'RNN', 4: 'DNN'}.get(int(model_type), 'UNKNOWN')

def evaluate_best_model(self, test='MLP'):
        """Évaluation complète du meilleur modèle sur le test set"""
        print(f"🎯 Évaluation du meilleur modèle {test}...")

        if _is_automl_mode(test):
            if getattr(self, 'best_params', None):
                cfg = dict(self.best_params)
                if 'optimizer' in cfg and 'optimizer_idx' not in cfg:
                    cfg['optimizer_idx'] = cfg['optimizer']
            else:
                cfg = _decode_automl(self.best_individual)
            model_name = _model_name_from_type(cfg['model_type'])
            print(f"🤖 AutoML best model selected: {model_name}")

            best_model, x_train, x_val = _build_automl_model(self, cfg)

            if cfg['model_type'] == 1:
                x_test = _reshape_cnn_input(self.X_test)
            elif cfg['model_type'] in (2, 3):
                x_test = _reshape_recurrent_input(self, self.X_test)
            else:
                x_test = self.X_test

            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )

            epochs = getattr(self, 'fixed_epochs', 100)

            y_train_fit = np.asarray(self.y_train).astype(int)
            y_val_fit = np.asarray(self.y_val).astype(int)

            history = best_model.fit(
                x_train, y_train_fit,
                validation_data=(x_val, y_val_fit),
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
            epochs = getattr(self, 'fixed_epochs', 100)
        elif test == 'MLP':
            best_model = create_mlp_model(self, n_dense_layers=self.best_individual[0], dense_units=self.best_individual[1:6], dropout_rate=self.best_individual[6], learning_rate=self.best_individual[7], optimizer_idx=self.best_individual[8], activation=self.best_individual[9])
            batch_size  = self.best_individual[10]
            epochs = getattr(self, 'fixed_epochs', 100)
        elif test == 'CNN':
            best_model = create_cnn_model(self, n_conv_layers=self.best_individual[0], conv_filters=self.best_individual[1:4], kernel_sizes=self.best_individual[4:7], pool_sizes=self.best_individual[7:10], n_dense_layers=self.best_individual[10], dense_units=self.best_individual[11:16], dropout_rate=self.best_individual[16], learning_rate=self.best_individual[17], optimizer_idx=self.best_individual[18], activation=self.best_individual[19])
            batch_size  = self.best_individual[20]
            epochs = getattr(self, 'fixed_epochs', 100)
        elif test == 'DNN':
            best_model = create_dnn_model(self, n_hidden_layers=self.best_individual[0], hidden_units=self.best_individual[1:6], dropout_rate=self.best_individual[6], learning_rate=self.best_individual[7], optimizer_idx=self.best_individual[8], activation=self.best_individual[9])
            batch_size = self.best_individual[10]
            epochs = getattr(self, 'fixed_epochs', 100)
        elif test == 'LSTM':
            best_model = create_lstm_model(self, n_lstm_layers=self.best_individual[0], lstm_units=self.best_individual[1:4], dropout_rate=self.best_individual[4], rec_dropout_rate=self.best_individual[5], n_dense_layers=self.best_individual[6], dense_units=self.best_individual[7:10], learning_rate=self.best_individual[10], optimizer_idx=self.best_individual[11], activation=self.best_individual[12])
            batch_size = self.best_individual[13]
            epochs = getattr(self, 'fixed_epochs', 100)
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        x_train_raw, x_val_raw, y_train_fit, y_val_fit = _get_training_and_validation_split(self)
        x_train_fit, x_val_fit = _prepare_train_val_for_model(self, test, x_train_raw, x_val_raw)
        x_test_fit = _reshape_cnn_input(self.X_test) if test == 'CNN' else _reshape_recurrent_input(self, self.X_test) if test in ('RNN', 'LSTM') else self.X_test
        # Entraînement complet
        history = best_model.fit(
            x_train_fit, y_train_fit,
            validation_data=(x_val_fit, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Prédictions sur test set
        y_pred = best_model.predict(x_test_fit, verbose=0, batch_size=1024)
        
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
                epochs = getattr(self, 'fixed_epochs', 100)
                early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0
                )

                x_train_raw, x_val_raw, y_train_fit, y_val_fit = _get_training_and_validation_split(self)
                model, x_train, x_val = _build_automl_model(self, cfg, X_train=x_train_raw, X_val=x_val_raw)

                history = model.fit(
                    x_train, y_train_fit,
                    validation_data=(x_val, y_val_fit),
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
                fitness = float(f1_score(y_val_fit, y_pred_classes, average='weighted'))

                del model, history, y_pred, y_pred_classes
                tf.keras.backend.clear_session()
                gc.collect()

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

                return (fitness,)

            # Création du modèle
            #model = create_mlp_model(self, individual[0],individual[1:6], individual[6] ,individual[7],individual[8] ,individual[9])
            def _build_model_for_test():
                if test == 'RNN':
                    return create_rnn_model(self, n_layers=individual[0], rnn_units=individual[1], n_dense=individual[2], dens=individual[3:8], optimizer_idx=individual[8], activation=individual[9], dropout_rate=individual[10], learning_rate=individual[11]), individual[12]
                if test == 'MLP':
                    return create_mlp_model(self, individual[0], individual[1:6], individual[6], individual[7], individual[8], individual[9]), individual[10]
                if test == 'CNN':
                    return create_cnn_model(self, n_conv_layers=individual[0], conv_filters=individual[1:4], kernel_sizes=individual[4:7], pool_sizes=individual[7:10], n_dense_layers=individual[10], dense_units=individual[11:16], dropout_rate=individual[16], learning_rate=individual[17], optimizer_idx=individual[18], activation=individual[19]), individual[20]
                if test == 'DNN':
                    return create_dnn_model(self, n_hidden_layers=individual[0], hidden_units=individual[1:6], dropout_rate=individual[6], learning_rate=individual[7], optimizer_idx=individual[8], activation=individual[9]), individual[10]
                if test == 'LSTM':
                    return create_lstm_model(self, n_lstm_layers=individual[0], lstm_units=individual[1:4], dropout_rate=individual[4], rec_dropout_rate=individual[5], n_dense_layers=individual[6], dense_units=individual[7:10], learning_rate=individual[10], optimizer_idx=individual[11], activation=individual[12]), individual[13]
                raise ValueError(f"Unsupported model test mode: {test}")

            epochs = getattr(self, 'fixed_epochs', 100)
                
            
            # Déterminer la taille de batch
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=0
            )

            x_train_raw, x_val_raw, y_train_fit, y_val_fit = _get_training_and_validation_split(self)
            model, batch_size = _build_model_for_test()
            x_train_fit, x_val_fit = _prepare_train_val_for_model(self, test, x_train_raw, x_val_raw)

            # Entraînement
            history = model.fit(
                x_train_fit, y_train_fit,
                validation_data=(x_val_fit, y_val_fit),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )

            # Prédictions sur validation
            y_pred = model.predict(x_val_fit, verbose=0, batch_size=1024)

            if self.n_classes == 2:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)

            # CALCUL DU F1 SCORE (NOUVELLE METRIQUE PRINCIPALE)
            f1 = f1_score(y_val_fit, y_pred_classes, average='weighted')

            # Nettoyage mémoire
            del model, history, y_pred, y_pred_classes
            tf.keras.backend.clear_session()
            gc.collect()

            return (f1,)
            
        except Exception as e:
            print(individual)
            print(f"⚠️ Erreur lors de l'évaluation: {str(e)}")
            return (0.0,)
