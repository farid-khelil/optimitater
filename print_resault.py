def _decode_best_params(individual, test='MLP'):
        """Compact decoder for legacy (non-AutoML) individuals."""
        optimizers_names = ['Adam', 'RMSprop', 'SGD']

        if test == 'MLP':
                n_dense = individual[0]
                return {
                        'model_name': 'MLP',
                        'n_dense_layers': n_dense,
                        'dense_units': individual[1:1+n_dense],
                        'dropout_rate': individual[6],
                        'learning_rate': individual[7],
                        'optimizer': optimizers_names[individual[8]],
                        'activation': individual[9],
                        'batch_size': individual[10],
                }

        if test == 'CNN':
                n_conv_layers = individual[0]
                n_dense_layers = individual[10]
                return {
                        'model_name': 'CNN',
                        'n_conv_layers': n_conv_layers,
                        'conv_filters': individual[1:1+n_conv_layers],
                        'kernel': individual[4:7],
                        'pool': individual[7:10],
                        'n_dense_layers': n_dense_layers,
                        'dense_units': individual[11:11+n_dense_layers],
                        'dropout_rate': individual[16],
                        'learning_rate': individual[17],
                        'optimizer': optimizers_names[individual[18]],
                        'activation': individual[19],
                        'batch_size': individual[20],
                }

        if test == 'RNN':
                n_layers = individual[0]
                n_dense = individual[2]
                return {
                        'model_name': 'RNN',
                        'n_layers': n_layers,
                        'rnn_units': individual[1],
                        'n_dense': n_dense,
                        'dense_units': individual[3:3+n_dense],
                        'dropout_rate': individual[10],
                        'learning_rate': individual[11],
                        'optimizer': optimizers_names[individual[8]],
                        'activation': individual[9],
                        'batch_size': individual[12],
                }

        if test == 'DNN':
                n_hidden_layers = individual[0]
                return {
                        'model_name': 'DNN',
                        'n_hidden_layers': n_hidden_layers,
                        'hidden_units': individual[1:1+n_hidden_layers],
                        'dropout_rate': individual[6],
                        'learning_rate': individual[7],
                        'optimizer': optimizers_names[individual[8]],
                        'activation': individual[9],
                        'batch_size': individual[10],
                }

        if test == 'LSTM':
                n_lstm_layers = individual[0]
                n_dense_layers = individual[6]
                return {
                        'model_name': 'LSTM',
                        'n_lstm_layers': n_lstm_layers,
                        'lstm_units': individual[1:1+n_lstm_layers],
                        'dropout_rate': individual[4],
                        'rec_dropout_rate': individual[5],
                        'n_dense_layers': n_dense_layers,
                        'dense_units': individual[7:7+n_dense_layers],
                        'learning_rate': individual[10],
                        'optimizer': optimizers_names[individual[11]],
                        'activation': individual[12],
                        'batch_size': individual[13],
                }

        return {'model_name': str(test), 'individual': list(individual)}


def _relevant_automl_params(params):
        """Keep only relevant fields for the selected model architecture."""
        model_name = params.get('model_name', 'UNKNOWN')
        out = {
                'model_type': params.get('model_type'),
                'model_name': model_name,
                'learning_rate': params.get('learning_rate'),
                'batch_size': params.get('batch_size'),
                'dropout_rate': params.get('dropout_rate'),
                'optimizer': params.get('optimizer'),
                'activation': params.get('activation'),
        }

        if model_name in ('MLP', 'DNN'):
                n_dense = int(params.get("n_dense_layers") or 1)
                dense_unit = int(params.get("dense_units") or 32)
                out['neurons'] = params.get('neurons')
                out['n_dense_layers'] = n_dense
                out['dense_units'] = [dense_unit] * n_dense
        elif model_name == 'CNN':
                n_conv = int(params.get("n_conv_layers") or 1)
                n_dense = int(params.get("n_dense_layers") or 1)
                dense_unit = int(params.get("dense_units") or 32)
                filters = int(params.get("filters") or 16)
                kernel = int(params.get("kernel_size") or 3)
                pool = int(params.get("pool_size") or 2)
                out['n_conv_layers'] = n_conv
                out['conv_filters'] = [filters] * n_conv
                out['kernel_sizes'] = [kernel] * n_conv
                out['pool_sizes'] = [pool] * n_conv
                out['n_dense_layers'] = n_dense
                out['dense_units'] = [dense_unit] * n_dense
        elif model_name == 'LSTM':
                n_dense = int(params.get("n_dense_layers") or 1)
                dense_unit = int(params.get("dense_units") or 32)
                units = int(params.get("units") or 32)
                n_lstm_layers = 1
                out['n_lstm_layers'] = n_lstm_layers
                out['lstm_units'] = [units] * n_lstm_layers
                out['units'] = units
                out['rec_dropout_rate'] = 0.1
                out['n_dense_layers'] = n_dense
                out['dense_units'] = [dense_unit] * n_dense
        elif model_name == 'RNN':
                n_dense = int(params.get("n_dense_layers") or 1)
                dense_unit = int(params.get("dense_units") or 32)
                units = int(params.get("units") or 32)
                n_rnn_layers = 1
                out['n_rnn_layers'] = n_rnn_layers
                out['rnn_units'] = [units] * n_rnn_layers
                out['units'] = units
                out['n_dense_layers'] = n_dense
                out['dense_units'] = [dense_unit] * n_dense

        return out


def display_results(self, execution_time,test='MLP',method='GA'):
        """Affichage des résultats finaux"""
        automl_mode = test in ('AUTOML', 'AUTO', 'ALL', 'UNIFIED')
        print("\n" + "="*80)
        print(f"🎊 RÉSULTATS DE L'OPTIMISATION {test} AVEC ALGORITHME {method}")
        print("="*80)
        if method == 'GA':
                print(f"🧬 PARAMÈTRES ALGORITHME GÉNÉTIQUE:")
                print(f"   • Générations: {self.generations}")
                print(f"   • Taille population: {self.population_size}")
                print(f"   • Probabilité croisement: {self.crossover_prob}")
                print(f"   • Probabilité mutation: {self.mutation_prob}")
                print(f"   • Temps d'exécution: {execution_time:.2f} secondes")
        elif method == 'GWO':
                print(f"🐺 PARAMÈTRES GRAY WOLF OPTIMIZER:")
                print(f"   • Temps d'exécution: {execution_time:.2f} secondes")
                if hasattr(self, 'gwo_tested_solutions'):
                        print(f"   • Solutions testées: {len(self.gwo_tested_solutions)}")
                if getattr(self, 'gwo_alpha', None) is not None:
                        print(f"   • Alpha score: {self.gwo_alpha.get('score', 'N/A')}")
                if getattr(self, 'gwo_beta', None) is not None:
                        print(f"   • Beta score: {self.gwo_beta.get('score', 'N/A')}")
                if getattr(self, 'gwo_delta', None) is not None:
                        print(f"   • Delta score: {self.gwo_delta.get('score', 'N/A')}")

        if automl_mode:
                print(f"\n🤖 RÉSUMÉ AUTOML (UN MODÈLE PAR TYPE):")
                if hasattr(self, '_automl_best_by_model') and self._automl_best_by_model is not None:
                        print(f"   • Types de modèles testés: {len(self._automl_best_by_model)}")
                if hasattr(self, 'best_fitness'):
                        print(f"   • Meilleur score validation (F1): {self.best_fitness:.4f}")
                print(f"   • Époques d'entraînement (fixes): {getattr(self, 'fixed_epochs', 100)}")

                if hasattr(self, 'top_k_results') and self.top_k_results:
                        best_item = self.top_k_results[0]
                        best_params_relevant = _relevant_automl_params(best_item.get('params', {}))

                        print("\n🧾 PARAMÈTRES DU MEILLEUR MODÈLE (PERTINENTS POUR L'ARCHITECTURE):")
                        for k, v in best_params_relevant.items():
                                print(f"   • {k}: {v}")
                        print("   • epochs (fixed): 100")

                        print("\n🏆 TOP RÉSULTATS (SANS RÉPÉTITION DE MODÈLE):")
                        for item in self.top_k_results:
                                p = _relevant_automl_params(item.get('params', {}))
                                print(
                                        f"   {item.get('rank', '?')}. {p.get('model_name', 'UNKNOWN')} | "
                                        f"fitness(F1)={item.get('fitness', 0.0):.4f} | "
                                        f"params={p}"
                                )
                else:
                        print("   • Aucun résultat AutoML à afficher.")

                if getattr(self, 'best_metrics', None):
                        print(f"\n📊 PERFORMANCE TEST (MEILLEUR MODÈLE):")
                        print(f"   • Accuracy: {self.best_metrics.get('accuracy', 0.0):.4f}")
                        print(f"   • Precision: {self.best_metrics.get('precision', 0.0):.4f}")
                        print(f"   • Recall: {self.best_metrics.get('recall', 0.0):.4f}")
                        print(f"   • F1-Score: {self.best_metrics.get('f1_score', 0.0):.4f}")

                print("="*80)
                return

        best_params = _decode_best_params(self.best_individual, test=test)
        print("\n🏆 MEILLEUR PARAMÉTRAGE OBTENU:")
        for k, v in best_params.items():
                print(f"   • {k}: {v}")
        print(f"   • epochs (fixed): {getattr(self, 'fixed_epochs', 100)}")

        print(f"\n📊 PERFORMANCE DU MEILLEUR MODÈLE:")
        print(f"   • Accuracy: {self.best_metrics.get('accuracy', 0.0):.4f}")
        print(f"   • Precision: {self.best_metrics.get('precision', 0.0):.4f}")
        print(f"   • Recall: {self.best_metrics.get('recall', 0.0):.4f}")
        print(f"   • F1-Score: {self.best_metrics.get('f1_score', 0.0):.4f}")

        print("="*80)
