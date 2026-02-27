import numpy as np
import tensorflow as tf
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from MLP import create_mlp_model


def evaluate_best_model(self):
        """Évaluation complète du meilleur modèle sur le test set"""
        print("🎯 Évaluation du meilleur modèle MLP...")
        
        # Création du meilleur modèle
        best_model = create_mlp_model(self, self.best_individual[0],self.best_individual[1:6], self.best_individual[6] ,self.best_individual[7],self.best_individual[8] ,self.best_individual[9])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        batch_sizes = [16, 32, 64, 128]
        batch_size  = batch_sizes[self.best_individual[10]]
        # Entraînement complet
        history = best_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=20,
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

def evaluate_individual(self, individual):
        """Évaluation d'un individu avec gestion mémoire"""
        try:
            # Création du modèle
            model = create_mlp_model(self, individual[0],individual[1:6], individual[6] ,individual[7],individual[8] ,individual[9])
            batch_sizes = [16, 32, 64, 128]
            batch_size = batch_sizes[individual[10]]
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
                epochs=30,
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
            print(f"⚠️ Erreur lors de l'évaluation: {str(e)}")
            return (0.0,)