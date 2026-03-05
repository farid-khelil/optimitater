
from time import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from CAE import build_cae_model
from GA import run_ga_optimization
from load_data import load_data
from load_data_salmi import load_and_preprocess_data
from GS import grid_search_optimization
from RS import randomized_search_optimization
from MLP import get_mlp_param, create_mlp_model
import os
from print_resault import display_results
from eval import evaluate_best_model
from compare_models import compare_all_models  # ← new: all-models comparison

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class MODEL:
    """
    Optimiseur professionnel MLP avec Algorithme Génétique (Version Portable)
    Optimise les hyperparamètres pour maximiser le recall
    """

    def __init__(self, data_path):
        """Initialisation avec le chemin du dataset"""

        self.saved_model_path = '/home/farid/pfe/models/saved/PremierLeague.keras'
        self.saved_data_path = '/home/farid/pfe/data/processed/PremierLeague_processed_data.csv'
        self.data_path = data_path
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.n_features = None
        self.n_classes = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.result_encoder = LabelEncoder()
        self.home_encoder = LabelEncoder()
        self.away_encoder = LabelEncoder()

        # Paramètres GA optimisés pour portabilité
        self.population_size = 2
        self.generations = 1
        self.crossover_prob = 0.85
        self.mutation_prob = 0.15

        # Meilleurs résultats
        self.best_individual = None
        self.best_fitness = 0
        self.best_metrics = {}


# obj = MODEL('/home/farid/pfe/data/Ransomware_headers.xlsx')
# obj = MODEL('/home/azureuser/cloudfiles/code/Users/faridklc17/Ransomware_headers.xlsx')
obj = MODEL('/home/azureuser/cloudfiles/code/Users/faridklc17/src/RBA.xlsx')
load_data(obj)
# load_and_preprocess_data(obj)

# model = build_cae_model(obj)
# model = create_mlp_model(obj=obj, n_dense_layers=1, dense_units=[64, 128, 256, 128, 64], dropout_rate=0.2, learning_rate=0.001, optimizer_idx=1, activation_idx=0, batch_size_idx=0)


# history = model.fit(
#     obj.X_train, obj.y_train,
#     validation_data=(obj.X_val, obj.y_val),
#     epochs=100,
#     batch_size=32,
# )

# ── run GA on ALL models, rank, chart & log ───────────────────────────────
compare_all_models(obj)

# ── single-model run (kept for reference, commented out) ──────────────────
# execution_time = run_ga_optimization(obj, test='CNN')
# evaluate_best_model(obj, test='CNN')
# display_results(obj, execution_time, test='CNN')
# randomized_search_optimization(obj)
