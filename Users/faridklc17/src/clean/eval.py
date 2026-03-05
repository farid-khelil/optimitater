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

def evaluate_best_model(self, test='MLP'):
        """Évaluation complète du meilleur modèle sur le test set"""
        print(f"🎯 Évaluation du meilleur modèle {test}...")
        
        # Création du meilleur modèle
        #best_model = create_mlp_model(self, self.best_individual[0],self.best_individual[1:6], self.best_individual[6] ,self.best_individual[7],self.best_individual[8] ,self.best_individual[9])
        if test == 'RNN':
            best_model = create_rnn_model(self, n_layers=self.best_individual[0], rnn_units=self.best_individual[1], n_dense=self.best_individual[2], dens=self.best_individual[3:8], optimizer_idx=self.best_individual[8], activation=self.best_individual[9], dropout_rate=self.best_individual[10], learning_rate=self.best_individual[11])
            batch_size  = self.best_individual[12]
            epochs = self.best_individual[13]
        elif test == 'MLP':
            best_model = create_mlp_model(self, n_dense_layers=self.best_individual[0], dense_units=self.best_individual[1:6], dropout_rate=self.best_individual[6], learning_rate=self.best_individual[7], optimizer_idx=self.best_individual[8], activation_idx=self.best_individual[9])
            batch_size  = self.best_individual[10]
            epochs = 100
        elif test == 'CNN':
            best_model = create_cnn_model(self, n_conv_layers=self.best_individual[0], conv_filters=self.best_individual[1:4], kernel_sizes=self.best_individual[4:7], pool_sizes=self.best_individual[7:10], n_dense_layers=self.best_individual[10], dense_units=self.best_individual[11:16], dropout_rate=self.best_individual[16], learning_rate=self.best_individual[17], optimizer_idx=self.best_individual[18], activation_idx=self.best_individual[19])
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
            # Création du modèle
            #model = create_mlp_model(self, individual[0],individual[1:6], individual[6] ,individual[7],individual[8] ,individual[9])
            if test == 'RNN':
                model = create_rnn_model(self, n_layers=individual[0], rnn_units=individual[1], n_dense=individual[2], dens=individual[3:8], optimizer_idx=individual[8], activation=individual[9], dropout_rate=individual[10], learning_rate=individual[11])
                batch_size  = individual[12]
                epochs = individual[13]
            elif test == 'MLP':
                model = create_mlp_model(self, individual[0], individual[1:6], individual[6], individual[7], individual[8], individual[9])
                batch_size  = individual[10]
            elif test == 'CNN':
                model = create_cnn_model(self, n_conv_layers=individual[0], conv_filters=individual[1:4], kernel_sizes=individual[4:7], pool_sizes=individual[7:10], n_dense_layers=individual[10], dense_units=individual[11:16], dropout_rate=individual[16], learning_rate=individual[17], optimizer_idx=individual[18], activation=individual[19])
                batch_size  = individual[20]
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