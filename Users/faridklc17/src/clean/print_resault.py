def display_results(self, execution_time):
        """Affichage des résultats finaux"""
        print("\n" + "="*80)
        print("🎊 RÉSULTATS DE L'OPTIMISATION MLP AVEC ALGORITHME GÉNÉTIQUE")
        print("="*80)
        
        # Paramètres GA
        print(f"🧬 PARAMÈTRES ALGORITHME GÉNÉTIQUE:")
        print(f"   • Générations: {self.generations}")
        print(f"   • Taille population: {self.population_size}")
        print(f"   • Probabilité croisement: {self.crossover_prob}")
        print(f"   • Probabilité mutation: {self.mutation_prob}")
        print(f"   • Temps d'exécution: {execution_time:.2f} secondes")
        
        # Meilleur paramétrage
        best_params = decode_individual(individual= self.best_individual)
        print(f"\n🏆 MEILLEUR PARAMÉTRAGE OBTENU:")
        print(f"   • Couches denses: {best_params['n_dense_layers']}")
        print(f"   • Unités par couche: {best_params['dense_units']}")
        print(f"   • Taux dropout: {best_params['dropout_rate']}")
        print(f"   • Taux apprentissage: {best_params['learning_rate']}")
        print(f"   • Optimiseur: {best_params['optimizer']}")
        print(f"   • Activation: {best_params['activation']}")
        print(f"   • Batch size: {best_params['batch_size']}")
        
        # Fitness et métriques
        print(f"\n📊 PERFORMANCE DU MEILLEUR MODÈLE:")
        print(f"   • Fitness (Recall): {self.best_fitness:.4f}")
        print(f"   • Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"   • Precision: {self.best_metrics['precision']:.4f}")
        print(f"   • Recall: {self.best_metrics['recall']:.4f}")
        print(f"   • F1-Score: {self.best_metrics['f1_score']:.4f}")
        
        print("="*80)


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
            'activation': activations_names[individual[9]],
            'batch_size': batch_sizes[individual[10]]
        }
        
        return params
