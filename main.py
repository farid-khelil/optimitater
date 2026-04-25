from time import time
import os
import sys
import random
import datetime
import atexit

# Set deterministic-related env vars as early as possible (before TF/model imports)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler ,MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from CAE import build_cae_model
from DNN import create_dnn_model
from GA import run_ga_optimization
from LSTM import create_lstm_model
from RNN import create_rnn_model
from load_data import load_data
from load_data_salmi import load_and_preprocess_data
from GS import grid_search_optimization
from RS import randomized_search_optimization
from MLP import get_mlp_param, create_mlp_model
from CNN import get_cnn_param, create_cnn_model
from print_resault import display_results
from eval import evaluate_best_model
from compare_models import compare_all_models
from GWO import GrayWolfOptimizer
# ── Environment detection ─────────────────────────────────────────────────
ON_KAGGLE   = os.path.isdir('/kaggle/working')
BASE_OUT    = '/kaggle/working'                  if ON_KAGGLE else '/home/farid/pfe'
DATA_ROOT   = '/kaggle/input'                    if ON_KAGGLE else '/home/farid/pfe/data/processed'
RISS_PATH   = f'{DATA_ROOT}/riss-dataset/RISS.csv' if ON_KAGGLE else f'{DATA_ROOT}/ransomware/WPD.xlsx'
# ──────────────────────────────────────────────────────────────────────────
SEED = 43

def set_global_determinism(seed=42):
    """Best-effort deterministic setup for reproducible runs."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    # Single-threaded ops can improve reproducibility across runs/hardware
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

set_global_determinism(SEED)
print(f"🎯 Reproducibility seed set to: {SEED}")

class _Tee:
    """Write to both the original stream and a log file simultaneously."""
    def __init__(obj, original, log_path):
        obj._orig = original
        obj._file = open(log_path, 'a', encoding='utf-8', buffering=1)
        obj._file.write(
            f"\n{'='*80}\n"
            f"  Session started: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n"
            f"{'='*80}\n"
        )
        atexit.register(obj.close)

    def write(obj, data):
        obj._orig.write(data)
        # Strip ANSI colour codes before writing to file
        import re
        clean = re.sub(r'\x1b\[[0-9;]*m', '', data)
        obj._file.write(clean)

    def flush(obj):
        obj._orig.flush()
        obj._file.flush()

    def close(obj):
        if not obj._file.closed:
            obj._file.write(
                f"\n{'='*80}\n"
                f"  Session ended:   {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n"
                f"{'='*80}\n"
            )
            obj._file.close()

    # Delegate everything else (isatty, fileno, …) to the original stream
    def __getattr__(obj, name):
        return getattr(obj._orig, name)

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
class MODEL:
    """
    Optimiseur professionnel MLP avec Algorithme Génétique (Version Portable)
    Optimise les hyperparamètres pour maximiser le recall
    """

    def __init__(obj, data_path):
        """Initialisation avec le chemin du dataset"""

        obj.saved_model_path = os.path.join(BASE_OUT, 'models', 'saved', 'PremierLeague.keras')
        obj.saved_data_path  = os.path.join(BASE_OUT, 'data', 'processed', 'PremierLeague_processed_data.csv')
        obj.data_path = data_path
        obj.X_train = None
        obj.X_val = None
        obj.X_test = None
        obj.y_train = None
        obj.y_val = None
        obj.y_test = None
        obj.n_features = None
        obj.n_classes = None
        obj.model = None
        obj.smootheringScaler = MinMaxScaler()
        obj.scaler = StandardScaler()
        obj.label_encoder = LabelEncoder()
        obj.result_encoder = LabelEncoder()
        obj.home_encoder = LabelEncoder()
        obj.away_encoder = LabelEncoder()

        # Paramètres GA optimisés pour portabilité
        obj.population_size = 5
        obj.generations = 0
        obj.crossover_prob = 0.85
        obj.mutation_prob = 0.15
        # Paramètres GWO optimisés pour portabilité
        obj.target_evaluations = 300
        obj.pop_size = 15
        # Meilleurs résultats
        obj.best_individual = None
        obj.best_fitness = 0
        obj.best_metrics = {}

        # CV config used by load_data/eval (80% train pool -> k-fold train/val)
        obj.use_cv = True
        obj.cv_folds = 5
        obj.cv_indices = None
        obj.X_train_val = None
        obj.y_train_val = None

def test_model(obj):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    preds = obj.model.predict(obj.X_test)

    # ── Decode predictions correctly based on output activation ──────────
    # sigmoid  → shape (N, 1), threshold at 0.5
    # softmax  → shape (N, C), take argmax
    if obj.n_classes == 2:                              # sigmoid output
        classes = (preds > 0.5).astype(int).flatten()
    else:                                                # softmax output
        classes = preds.argmax(axis=1)

    y_true = np.array(obj.y_test).flatten()

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
# CV settings (80% training pool split into k folds; each fold is ~80/20 train/val)
obj.use_cv = True
obj.cv_folds = 5
load_data(obj, idx='2')
# load_and_preprocess_data(obj)

obj.model = create_cnn_model(
    obj=obj,
    n_conv_layers=3,
    conv_filters=[23, 23, 23],
    kernel_sizes=[3, 3, 3],
    pool_sizes=[2, 2, 2],
    n_dense_layers=2,
    dense_units=[32, 32],
    dropout_rate=0.2,
    learning_rate=0.0031144456644091895,
    optimizer_idx=0,
    activation='leaky_relu'
)
# GrayWolfOptimizer(obj,test='MLP', target_evaluations=obj.target_evaluations, pop_size=obj.pop_size)
# execution_time = run_ga_optimization(obj, test='AUTOML')
# evaluate_best_model(obj, test='AUTOML')
# display_results(obj, execution_time, test='AUTOML', method='GA')

obj .model = create_mlp_model(
    obj=obj,
    optimizer_idx=0,
    activation='sigmoid',
    dropout_rate=0.2,
    learning_rate=0.0024082891265896104,
    n_dense_layers=3,
    dense_units=[128, 128, 128]
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = obj.model.fit(
    obj.X_train, obj.y_train,
    validation_data=(obj.X_val, obj.y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
)
test_model(obj)
# ── run GA on ALL models, rank, chart & log ───────────────────────────────
# compare_all_models(obj)

# ── single-model run (kept for reference, commented out) ──────────────────
# execution_time = run_ga_optimization(obj, test='CNN')
# evaluate_best_model(obj, test='CNN')
# display_results(obj, execution_time, test='CNN')
# randomized_search_optimization(obj)
