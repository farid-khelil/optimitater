def display_results(self, execution_time,test='MLP'):
        """Affichage des résultats finaux"""
        print("\n" + "="*80)
        print(f"🎊 RÉSULTATS DE L'OPTIMISATION {test} AVEC ALGORITHME GÉNÉTIQUE")
        print("="*80)
        
        # Paramètres GA
        print(f"🧬 PARAMÈTRES ALGORITHME GÉNÉTIQUE:")
        print(f"   • Générations: {self.generations}")
        print(f"   • Taille population: {self.population_size}")
        print(f"   • Probabilité croisement: {self.crossover_prob}")
        print(f"   • Probabilité mutation: {self.mutation_prob}")
        print(f"   • Temps d'exécution: {execution_time:.2f} secondes")
        
        # Meilleur paramétrage
        if test == 'MLP':
            best_params = decode_individual(individual= self.best_individual)
            print(f"\n🏆 MEILLEUR PARAMÉTRAGE OBTENU:")
            print(f"   • Couches denses: {best_params['n_dense_layers']}")
            print(f"   • Unités par couche: {best_params['dense_units']}")
            print(f"   • Taux dropout: {best_params['dropout_rate']}")
            print(f"   • Taux apprentissage: {best_params['learning_rate']}")
            print(f"   • Optimiseur: {best_params['optimizer']}")
            print(f"   • Activation: {best_params['activation']}")
            print(f"   • Batch size: {best_params['batch_size']}")
        elif test == 'CNN':
                best_params = decode_cnn_individual(individual= self.best_individual)
                print(f"\n🏆 MEILLEUR PARAMÉTRAGE OBTENU:")
                print(f"   • Couches convolutionnelles: {best_params['n_conv_layers']}")
                print(f"   • Filtres par couche: {best_params['conv_filters']}")
                print(f"   • Taille des noyaux: {best_params['kernel']}")
                print(f"   • Taille des pooling: {best_params['pool']}")
                print(f"   • Couches denses: {best_params['n_dense_layers']}")
                print(f"   • Unités par couche: {best_params['dense_units']}")
                print(f"   • Taux dropout: {best_params['dropout_rate']}")
                print(f"   • Taux apprentissage: {best_params['learning_rate']}")
                print(f"   • Optimiseur: {best_params['optimizer']}")
                print(f"   • Activation: {best_params['activation']}")
                print(f"   • Batch size: {best_params['batch_size']}")
                print(f"   • Époques: {best_params['epochs']}")
        elif test == 'RNN':
                best_params = decode_rnn_individual(individual= self.best_individual)
                print(f"\n🏆 MEILLEUR PARAMÉTRAGE OBTENU:")
                print(f"   • Couches RNN: {best_params['n_layers']}")
                print(f"   • Unités RNN: {best_params['rnn_units']}")
                print(f"   • Couches denses: {best_params['n_dense']}")
                print(f"   • Unités par couche dense: {best_params['dense_units']}")
                print(f"   • Taux dropout: {best_params['dropout_rate']}")
                print(f"   • Taux apprentissage: {best_params['learning_rate']}")
                print(f"   • Optimiseur: {best_params['optimizer']}")
                print(f"   • Activation: {best_params['activation']}")
                print(f"   • Batch size: {best_params['batch_size']}")
                print(f"   • Époques: {best_params['epochs']}")
        elif test == 'DNN':
                best_params = decode_dnn_individual(individual= self.best_individual)
                print(f"\n🏆 MEILLEUR PARAMÉTRAGE OBTENU:")
                print(f"   • Couches cachées: {best_params['n_hidden_layers']}")
                print(f"   • Unités par couche cachée: {best_params['hidden_units']}")
                print(f"   • Taux dropout: {best_params['dropout_rate']}")
                print(f"   • Taux apprentissage: {best_params['learning_rate']}")
                print(f"   • Optimiseur: {best_params['optimizer']}")
                print(f"   • Activation: {best_params['activation']}")
                print(f"   • Batch size: {best_params['batch_size']}")
                print(f"   • Époques: {best_params['epochs']}")
        elif test == 'LSTM':
                best_params = decode_lstm_individual(individual= self.best_individual)
                print(f"\n🏆 MEILLEUR PARAMÉTRAGE OBTENU:")
                print(f"   • Couches LSTM: {best_params['n_lstm_layers']}")
                print(f"   • Unités par couche LSTM: {best_params['lstm_units']}")
                print(f"   • Taux dropout: {best_params['dropout_rate']}")
                print(f"   • Taux dropout récurrent: {best_params['rec_dropout_rate']}")
                print(f"   • Couches denses: {best_params['n_dense_layers']}")
                print(f"   • Unités par couche dense: {best_params['dense_units']}")
                print(f"   • Taux apprentissage: {best_params['learning_rate']}")
                print(f"   • Optimiseur: {best_params['optimizer']}")
                print(f"   • Activation: {best_params['activation']}")
                print(f"   • Batch size: {best_params['batch_size']}")
                print(f"   • Époques: {best_params['epochs']}")
        # Fitness et métriques
        print(f"\n📊 PERFORMANCE DU MEILLEUR MODÈLE:")
        print(f"   • Fitness (Recall): {self.best_fitness:.4f}")
        print(f"   • Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"   • Precision: {self.best_metrics['precision']:.4f}")
        print(f"   • Recall: {self.best_metrics['recall']:.4f}")
        print(f"   • F1-Score: {self.best_metrics['f1_score']:.4f}")
        
        print("="*80)

def decode_rnn_individual(individual):
        """Décodage des paramètres de l'individu pour RNN"""
        optimizers_names = ['Adam', 'RMSprop', 'SGD']
        batch_sizes = [16, 32, 64, 128]
        
        n_layers = individual[0]
        n_dense = individual[2]
        
        params = {
            'n_layers': n_layers,
            'rnn_units': individual[1],
            'n_dense': n_dense,
            'dense_units': individual[3:3+n_dense],
            'dropout_rate': individual[10],
            'learning_rate': individual[11],
            'optimizer': optimizers_names[individual[8]],
            'activation': individual[9],
            'batch_size': individual[12],
            'epochs': individual[13]
        }
        return params
def decode_dnn_individual(individual):
        """Décodage des paramètres de l'individu pour DNN"""
        optimizers_names = ['Adam', 'RMSprop', 'SGD']
        
        n_hidden_layers = individual[0]
        
        params = {
            'n_hidden_layers': n_hidden_layers,
            'hidden_units': individual[1:1+n_hidden_layers],
            'dropout_rate': individual[6],
            'learning_rate': individual[7],
            'optimizer': optimizers_names[individual[8]],
            'activation': individual[9],   # already a string, not an index
            'batch_size': individual[10],
            'epochs': individual[11]
        }
        
        return params
def decode_lstm_individual(individual):
        """Décodage des paramètres de l'individu pour LSTM"""
        optimizers_names = ['Adam', 'RMSprop', 'SGD']
        
        n_lstm_layers = individual[0]
        n_dense_layers = individual[6]
        
        params = {
            'n_lstm_layers': n_lstm_layers,
            'lstm_units': individual[1:4],
            'dropout_rate': individual[4],
            'rec_dropout_rate': individual[5],
            'n_dense_layers': n_dense_layers,
            'dense_units': individual[7:10],
            'learning_rate': individual[10],
            'optimizer': optimizers_names[individual[11]],
            'activation': individual[12],
            'batch_size': individual[13],
            'epochs':individual[14]
        }
        
        return params

def decode_individual(individual):
        """Décodage des paramètres de l'individu pour MLP"""
        optimizers_names = ['Adam', 'RMSprop', 'SGD']
        activations_names = ['relu', 'elu', 'selu', 'tanh']
        batch_sizes = [16, 32, 64, 128]
        
        n_dense = individual[0]
        
        params = {
            'n_dense_layers': n_dense,
            'dense_units': individual[1:1+n_dense],
            'dropout_rate': individual[6],
            'learning_rate': individual[7],
            'optimizer': optimizers_names[individual[8]],
            'activation': individual[9],
            'batch_size': individual[10]
        }
        
        return params


def decode_cnn_individual(individual):
        """Décodage des paramètres de l'individu pour MLP"""
        optimizers_names = ['Adam', 'RMSprop', 'SGD']
        
        n_dense = individual[0]
        
        params = {
             'n_conv_layers':individual[0],
            'conv_filters': individual[1:individual[0]+1],
            'kernel':individual[4:7],
            'pool':individual[7:10],
            'n_dense_layers': individual[10],
            'dense_units':individual[11:16],
            'dropout_rate': individual[16],
            'learning_rate': individual[17],
            'optimizer': optimizers_names[individual[18]],
            'activation': individual[19],
            'batch_size': individual[20],
            'epochs':individual[21]
        }
        
        return params
