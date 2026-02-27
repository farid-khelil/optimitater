
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(obj):
        """Chargement et préparation des données optimisé pour mémoire"""
        print("📊 Chargement et préparation des données...")
        
        try:
            # Chargement du dataset
            df = pd.read_excel(obj.data_path)
            print(f"✅ Dataset chargé: {df.shape[0]} échantillons, {df.shape[1]} caractéristiques")
            
            # Correction des noms de colonnes
            df.columns = df.columns.astype(str)
            
            # Séparation features/target
            X = df.drop(['family'], axis=1)
            y = df['family']
            
            # Sélection des colonnes numériques
            numeric_columns = []
            for col in X.columns:
                try:
                    pd.to_numeric(X[col], errors='raise')
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    continue
            
            X = X[numeric_columns]
            
            if X.empty:
                raise ValueError("Aucune colonne numérique trouvée dans le dataset")
            
            # Encodage des labels
            y_encoded = obj.label_encoder.fit_transform(y)
            obj.n_classes = len(np.unique(y_encoded))
            
            print(f"📈 Nombre de classes: {obj.n_classes}")
            print(f"🔢 Caractéristiques numériques: {X.shape[1]}")
            
            # Conversion en numpy arrays
            X_values = X.values.astype(np.float32)
            
            # Split des données (70/20/10)
            X_train_val, obj.X_test, y_train_val, obj.y_test = train_test_split(
                X_values, y_encoded, test_size=0.10, random_state=42, stratify=y_encoded
            )
            
            obj.X_train, obj.X_val, obj.y_train, obj.y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2222, random_state=42, stratify=y_train_val
            )
            
            # Libération mémoire
            del df, X, X_values, X_train_val, y_train_val
            gc.collect()
            
            # Normalisation APRÈS split
            obj.X_train = obj.scaler.fit_transform(obj.X_train)
            obj.X_val = obj.scaler.transform(obj.X_val)
            obj.X_test = obj.scaler.transform(obj.X_test)
            
            # Pas de reshape nécessaire pour MLP (format 2D)
            obj.n_features = obj.X_train.shape[1]
            
            print(f"🎯 Données d'entraînement: {obj.X_train.shape} (70%)")
            print(f"🎯 Données de validation: {obj.X_val.shape} (20%)")
            print(f"🎯 Données de test: {obj.X_test.shape} (10%)")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données: {str(e)}")
            raise





