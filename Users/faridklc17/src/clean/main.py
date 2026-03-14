
from time import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from CAE import build_cae_model
from GA import run_ga_optimization
from load_data import load_data
from load_data_salmi import load_and_preprocess_data
from GS import grid_search_optimization
from RS import randomized_search_optimization
from MLP import get_mlp_param, create_mlp_model
from CNN import get_cnn_param, create_cnn_model
import os, sys
from print_resault import display_results
from eval import evaluate_best_model
from compare_models import compare_all_models  # ← new: all-models comparison
import numpy as np
from sklearn.metrics import accuracy_score

# ── Environment detection ─────────────────────────────────────────────────
import os, sys
ON_KAGGLE   = os.path.isdir('/kaggle/working')
BASE_OUT    = '/kaggle/working'                  if ON_KAGGLE else '/home/farid/pfe'
DATA_ROOT   = '/kaggle/input'                    if ON_KAGGLE else '/home/farid/pfe/data/processed'
RISS_PATH   = f'{DATA_ROOT}/riss-dataset/RISS.csv' if ON_KAGGLE else f'{DATA_ROOT}/ransomware/RISS.csv'
# ──────────────────────────────────────────────────────────────────────────

class _Tee:
    """Write to both the original stream and a log file simultaneously."""
    def __init__(self, original, log_path):
        self._orig = original
        self._file = open(log_path, 'a', encoding='utf-8', buffering=1)
        self._file.write(
            f"\n{'='*80}\n"
            f"  Session started: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n"
            f"{'='*80}\n"
        )
        atexit.register(self.close)

    def write(self, data):
        self._orig.write(data)
        # Strip ANSI colour codes before writing to file
        import re
        clean = re.sub(r'\x1b\[[0-9;]*m', '', data)
        self._file.write(clean)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.write(
                f"\n{'='*80}\n"
                f"  Session ended:   {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n"
                f"{'='*80}\n"
            )
            self._file.close()

    # Delegate everything else (isatty, fileno, …) to the original stream
    def __getattr__(self, name):
        return getattr(self._orig, name)

_LOG_PATH = os.path.join(BASE_OUT, 'results', 'run.log')
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
sys.stdout = _Tee(sys.stdout, _LOG_PATH)
sys.stderr = _Tee(sys.stderr, _LOG_PATH)
print(f"📄  Logging to: {_LOG_PATH}")
# ──────────────────────────────────────────────────────────────────────────

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    END = '\033[0m' # Reset color
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class MODEL:
    """
    Optimiseur professionnel MLP avec Algorithme Génétique (Version Portable)
    Optimise les hyperparamètres pour maximiser le recall
    """

    def __init__(self, data_path):
        """Initialisation avec le chemin du dataset"""

        self.saved_model_path = os.path.join(BASE_OUT, 'models', 'saved', 'PremierLeague.keras')
        self.saved_data_path  = os.path.join(BASE_OUT, 'data', 'processed', 'PremierLeague_processed_data.csv')
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

def test_model(self):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    preds = self.model.predict(self.X_test)

    # ── Decode predictions correctly based on output activation ──────────
    # sigmoid  → shape (N, 1), threshold at 0.5
    # softmax  → shape (N, C), take argmax
    if self.n_classes == 2:                              # sigmoid output
        classes = (preds > 0.5).astype(int).flatten()
    else:                                                # softmax output
        classes = preds.argmax(axis=1)

    y_true = np.array(self.y_test).flatten()

    acc  = accuracy_score(y_true, classes)
    prec = precision_score(y_true, classes, average='weighted', zero_division=0)
    rec  = recall_score(y_true, classes,    average='weighted', zero_division=0)
    f1   = f1_score(y_true, classes,        average='weighted', zero_division=0)
    cm   = confusion_matrix(y_true, classes)

    print(f"\n{Color.BLUE}── Test-set evaluation ──────────────────────{Color.END}")
    print(f"  pred shape : {preds.shape}  |  unique preds : {np.unique(classes)}")
    print(f"  {Color.GREEN}Accuracy   : {acc  * 100:.2f}%{Color.END}")
    print(f"  Precision  : {prec * 100:.2f}%")
    print(f"  Recall     : {rec  * 100:.2f}%")
    print(f"  F1-Score   : {f1   * 100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")



# obj = MODEL('/home/farid/pfe/data/Ransomware_headers.xlsx')
# obj = MODEL('/home/azureuser/cloudfiles/code/Users/faridklc17/Ransomware_headers.xlsx')
# obj = MODEL('/home/azureuser/cloudfiles/code/Users/faridklc17/src/RBA.xlsx')
obj = MODEL(RISS_PATH)
load_data(obj, idx='4')
# load_and_preprocess_data(obj)

# model = build_cae_model(obj)
# 3 layers: 256 → 128 → 64  (consistent n_dense_layers=3 / dense_units length)
# obj.model = create_cnn_model(
#     obj=obj,
# )

# from tensorflow.keras.callbacks import EarlyStopping
# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     restore_best_weights=True,
#     verbose=1
# )

# history = obj.model.fit(
#     obj.X_train, obj.y_train,
#     validation_data=(obj.X_val, obj.y_val),
#     epochs=100,
#     batch_size=32,
#     callbacks=[early_stop],
# )
# test_model(obj)
# ── run GA on ALL models, rank, chart & log ───────────────────────────────
# compare_all_models(obj)

# ── single-model run (kept for reference, commented out) ──────────────────
# execution_time = run_ga_optimization(obj, test='CNN')
# evaluate_best_model(obj, test='CNN')
# display_results(obj, execution_time, test='CNN')
# randomized_search_optimization(obj)
