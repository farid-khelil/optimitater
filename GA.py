
import time
from deap import creator, base, tools, algorithms
import random
import os
import tensorflow as tf
import gc
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from eval import evaluate_individual


def _set_global_seed(seed=None):
    """Set Python/NumPy/TensorFlow seeds for reproducible optimization runs."""
    if seed is None:
        return

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Unified AutoML search space
# Chromosome:
# [model_type, learning_rate, batch_size, epochs,
#  neurons, filters, kernel_size, units,
#  n_conv_layers, pool_size, n_dense_layers, dense_units,
#  dropout_rate, optimizer, activation]
# model_type: 0=MLP, 1=CNN, 2=LSTM, 3=RNN, 4=DNN
# ─────────────────────────────────────────────────────────────────────────────
MODEL_TYPE_MAP = {
    0: "MLP",
    1: "CNN",
    2: "LSTM",
    3: "RNN",
    4: "DNN",
}

AUTO_BATCH_SIZES = [16, 32, 64, 128]
AUTO_KERNEL_SIZES = [3, 5, 7]

AUTO_BOUNDS = {
    "learning_rate": (1e-5, 1e-2),
    "epochs": (100, 100),
    "neurons": (16, 512),
    "filters": (16, 256),
    "units": (16, 512),
}


def _clip(value, low, high):
    return max(low, min(high, value))


def decode_automl_individual(individual):
    optimizers_names = ['Adam', 'RMSprop', 'SGD']
    return {
        "model_type": int(individual[0]),
        "model_name": MODEL_TYPE_MAP.get(int(individual[0]), "UNKNOWN"),
        "learning_rate": float(individual[1]),
        "batch_size": int(individual[2]),
        "neurons": int(individual[4]),
        "filters": int(individual[5]),
        "kernel_size": int(individual[6]),
        "units": int(individual[7]),
        "n_conv_layers": int(individual[8]),
        "pool_size": int(individual[9]),
        "n_dense_layers": int(individual[10]),
        "dense_units": int(individual[11]),
        "dropout_rate": float(individual[12]),
        "optimizer": optimizers_names[int(individual[13])],
        "activation": individual[14],
    }


def _automl_relevant_params(params):
    """Return only model-relevant params for display."""
    model_name = params.get("model_name", "UNKNOWN")
    out = {
        "model_type": params.get("model_type"),
        "model_name": model_name,
        "learning_rate": params.get("learning_rate"),
        "batch_size": params.get("batch_size"),
        "dropout_rate": params.get("dropout_rate"),
        "optimizer": params.get("optimizer"),
        "activation": params.get("activation"),
    }

    if model_name in ("MLP", "DNN"):
        n_dense = int(params.get("n_dense_layers") or 1)
        dense_unit = int(params.get("dense_units") or 32)
        out["neurons"] = params.get("neurons")
        out["n_dense_layers"] = n_dense
        out["dense_units"] = [dense_unit] * n_dense
    elif model_name == "CNN":
        n_conv = int(params.get("n_conv_layers") or 1)
        n_dense = int(params.get("n_dense_layers") or 1)
        dense_unit = int(params.get("dense_units") or 32)
        filters = int(params.get("filters") or 16)
        kernel = int(params.get("kernel_size") or 3)
        pool = int(params.get("pool_size") or 2)
        out["n_conv_layers"] = n_conv
        out["conv_filters"] = [filters] * n_conv
        out["kernel_sizes"] = [kernel] * n_conv
        out["pool_sizes"] = [pool] * n_conv
        out["n_dense_layers"] = n_dense
        out["dense_units"] = [dense_unit] * n_dense
    elif model_name == "LSTM":
        n_dense = int(params.get("n_dense_layers") or 1)
        dense_unit = int(params.get("dense_units") or 32)
        units = int(params.get("units") or 32)
        n_lstm_layers = 1
        out["n_lstm_layers"] = n_lstm_layers
        out["lstm_units"] = [units] * n_lstm_layers
        out["units"] = units
        out["rec_dropout_rate"] = 0.1
        out["n_dense_layers"] = n_dense
        out["dense_units"] = [dense_unit] * n_dense
    elif model_name == "RNN":
        n_dense = int(params.get("n_dense_layers") or 1)
        dense_unit = int(params.get("dense_units") or 32)
        units = int(params.get("units") or 32)
        n_rnn_layers = 1
        out["n_rnn_layers"] = n_rnn_layers
        out["rnn_units"] = [units] * n_rnn_layers
        out["units"] = units
        out["n_dense_layers"] = n_dense
        out["dense_units"] = [dense_unit] * n_dense

    return out


def get_automl_param(toolbox):
    toolbox.register("model_type", random.randint, 0, 4)
    toolbox.register("learning_rate", random.uniform, AUTO_BOUNDS["learning_rate"][0], AUTO_BOUNDS["learning_rate"][1])
    toolbox.register("batch_size", random.choice, AUTO_BATCH_SIZES)
    toolbox.register("epochs", lambda: 100)
    toolbox.register("neurons", random.randint, AUTO_BOUNDS["neurons"][0], AUTO_BOUNDS["neurons"][1])
    toolbox.register("filters", random.randint, AUTO_BOUNDS["filters"][0], AUTO_BOUNDS["filters"][1])
    toolbox.register("kernel_size", random.choice, AUTO_KERNEL_SIZES)
    toolbox.register("units", random.randint, AUTO_BOUNDS["units"][0], AUTO_BOUNDS["units"][1])
    toolbox.register("n_conv_layers", random.choice, [1, 2, 3])
    toolbox.register("pool_size", random.choice, [2, 4])
    toolbox.register("n_dense_layers", random.choice, [1, 2, 3])
    toolbox.register("dense_units", random.choice, [32, 64, 128, 256])
    toolbox.register("dropout_rate", random.choice, [0.0, 0.1, 0.2, 0.3, 0.5])
    toolbox.register("optimizer", random.randint, 0, 2)
    toolbox.register("activation", random.choice, ['relu', 'tanh', 'sigmoid', 'leaky_relu'])

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.model_type,
            toolbox.learning_rate,
            toolbox.batch_size,
            toolbox.epochs,
            toolbox.neurons,
            toolbox.filters,
            toolbox.kernel_size,
            toolbox.units,
            toolbox.n_conv_layers,
            toolbox.pool_size,
            toolbox.n_dense_layers,
            toolbox.dense_units,
            toolbox.dropout_rate,
            toolbox.optimizer,
            toolbox.activation,
        ),
        n=1,
    )


def get_mlp_param(toolbox):
    toolbox.register("n_dense_layers", random.randint, 1, 5)
    toolbox.register("dense_units", random.choice, [32, 64, 128, 256, 512])
    toolbox.register("dropout_rate", random.choice, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    toolbox.register("learning_rate", random.choice, [0.0001, 0.0005, 0.001, 0.005, 0.01])
    toolbox.register("optimizer", random.randint, 0, 2)
    toolbox.register("activation", random.choice, ['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    toolbox.register("batch_size", random.choice, [16, 32, 64, 128])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.n_dense_layers, 
                         toolbox.dense_units, toolbox.dense_units, toolbox.dense_units, toolbox.dense_units, toolbox.dense_units,
                         toolbox.dropout_rate,
                         toolbox.learning_rate,
                         toolbox.optimizer,
                         toolbox.activation,
                         toolbox.batch_size), n=1)
def get_cnn_param(toolbox):
    toolbox.register("n_conv_layers", random.choice, [1, 2, 3])
    toolbox.register("conv_filters", random.choice, [16, 32, 64, 128])
    toolbox.register("kernel_sizes", random.choice, [3, 5, 7])
    toolbox.register("pool_sizes", random.choice, [2, 4])
    toolbox.register("n_dense_layers", random.choice, [1, 2, 3])
    toolbox.register("dense_units", random.choice, [32, 64, 128, 256])
    toolbox.register("dropout_rate", random.choice, [0.0, 0.1, 0.2, 0.3, 0.5])
    toolbox.register("learning_rate", random.choice, [0.0001, 0.0005, 0.001, 0.005, 0.01])
    toolbox.register("optimizer", random.randint, 0, 2)
    toolbox.register("activation", random.choice, ['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    toolbox.register("batch_size", random.choice,[16, 32, 64, 128])
    toolbox.register("n_epochs", random.choice, [100])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.n_conv_layers,
                         toolbox.conv_filters,toolbox.conv_filters,toolbox.conv_filters,
                         toolbox.kernel_sizes,toolbox.kernel_sizes,toolbox.kernel_sizes,
                         toolbox.pool_sizes,toolbox.pool_sizes,toolbox.pool_sizes,
                         toolbox.n_dense_layers,
                         toolbox.dense_units,toolbox.dense_units,toolbox.dense_units,toolbox.dense_units,toolbox.dense_units,
                         toolbox.dropout_rate,
                         toolbox.learning_rate,
                         toolbox.optimizer,
                         toolbox.activation,
                         toolbox.batch_size,
                         toolbox.n_epochs), n=1)

def get_rnn_param(toolbox):
    toolbox.register("n_rnn_layers", random.choice, [1, 2, 3])
    toolbox.register("rnn_units", random.choice, [64, 128, 256])
    toolbox.register("n_dense_layers", random.choice, [1, 2, 3, 4, 5])
    toolbox.register("dense_units", random.choice, [64, 128, 256, 512])
    toolbox.register("dropout_rate", random.choice, [0.0, 0.1, 0.2, 0.3, 0.5])
    toolbox.register("learning_rate", random.choice, [0.0001, 0.001, 0.005, 0.01, 0.05])
    toolbox.register("optimizer", random.randint, 0, 2)
    toolbox.register("activation", random.choice, ['relu', 'elu', 'selu', 'tanh'])
    toolbox.register("batch_size", random.choice, [16, 32, 64, 128])
    toolbox.register("n_epochs", random.choice, [100])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.n_rnn_layers,
                         toolbox.rnn_units,
                         toolbox.n_dense_layers,
                         toolbox.dense_units, toolbox.dense_units, toolbox.dense_units, toolbox.dense_units, toolbox.dense_units,
                         toolbox.optimizer,
                         toolbox.activation,
                         toolbox.dropout_rate,
                         toolbox.learning_rate,
                         toolbox.batch_size,
                         toolbox.n_epochs), n=1)
def get_dnn_param(toolbox):
    toolbox.register("n_hidden_layers", random.choice, [1, 2, 3, 4, 5])
    toolbox.register("hidden_units", random.choice, [32, 64, 128, 256, 512])
    toolbox.register("dropout_rate", random.choice, [0.0, 0.1, 0.2, 0.3, 0.5])
    toolbox.register("learning_rate", random.choice, [0.0001, 0.0005, 0.001, 0.005, 0.01])
    toolbox.register("optimizer_idx", random.choice, [0, 1, 2])
    toolbox.register("activation", random.choice, ['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    toolbox.register("batch_size", random.choice, [16, 32, 64])
    toolbox.register("n_epochs", random.choice, [100])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.n_hidden_layers,
                         toolbox.hidden_units, toolbox.hidden_units, toolbox.hidden_units, toolbox.hidden_units, toolbox.hidden_units,
                         toolbox.dropout_rate,
                         toolbox.learning_rate,
                         toolbox.optimizer_idx,
                         toolbox.activation,
                         toolbox.batch_size,
                         toolbox.n_epochs), n=1)


def get_lstm_param(toolbox):
    toolbox.register("n_lstm_layers", random.choice, [1, 2, 3])
    toolbox.register("lstm_units", random.choice, [32, 64, 128, 256])
    toolbox.register("dropout_rate", random.choice, [0.0, 0.1, 0.2, 0.3, 0.5])
    toolbox.register("rec_dropout_rate", random.choice, [0.0, 0.1, 0.2])
    toolbox.register("n_dense_layers", random.choice, [1, 2, 3])
    toolbox.register("dense_units", random.choice, [32, 64, 128, 256])
    toolbox.register("learning_rate", random.choice, [0.0001, 0.0005, 0.001, 0.005, 0.01])
    toolbox.register("optimizer_idx", random.choice, [0, 1, 2])
    toolbox.register("activation", random.choice, ['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    toolbox.register("batch_size", random.choice, [16, 32, 64])
    toolbox.register("n_epochs", random.choice, [100])
    toolbox.register("individual", tools.initCycle, creator.Individual,
                                (toolbox.n_lstm_layers,
                                 toolbox.lstm_units, toolbox.lstm_units, toolbox.lstm_units,
                                 toolbox.dropout_rate,
                                 toolbox.rec_dropout_rate,
                                 toolbox.n_dense_layers,
                                 toolbox.dense_units, toolbox.dense_units, toolbox.dense_units,
                                 toolbox.learning_rate,
                                 toolbox.optimizer_idx,
                                 toolbox.activation,
                                 toolbox.batch_size,
                                 toolbox.n_epochs), n=1)
    
def setup_genetic_algorithm(self, test='AUTOML'):
        """Configuration de l'AG sans parallélisme"""
        # Création des types  (guard: re-creating them crashes DEAP on second call)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        if test in ('AUTOML', 'AUTO', 'ALL', 'UNIFIED'):
            get_automl_param(toolbox)
        elif test == 'RNN':
            get_rnn_param(toolbox)
        elif test == 'MLP':
            get_mlp_param(toolbox)
        elif test == 'CNN':
            get_cnn_param(toolbox)
        elif test == 'DNN':
            get_dnn_param(toolbox)
        elif test == 'LSTM':
            get_lstm_param(toolbox)
        
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Utilisation du map standard (séquentiel)
        toolbox.register("map", map)
        
        toolbox.register("evaluate", evaluate_individual, self, test=test)
        toolbox.register("mate", tools.cxTwoPoint)
        # toolbox.evaluate()
        if test in ('AUTOML', 'AUTO', 'ALL', 'UNIFIED'):
            toolbox.register("mutate", custom_automl_mutation, self)
        elif test == 'RNN':
            toolbox.register("mutate", custom_rnn_mutation, self)
        elif test == 'MLP':
            toolbox.register("mutate", custom_mlp_mutation, self)
        elif test == 'CNN':
            toolbox.register("mutate", custom_cnn_mutation, self)
        elif test == 'DNN':
            toolbox.register("mutate", custom_dnn_mutation, self)
        elif test == 'LSTM':
            toolbox.register("mutate", custom_lstm_mutation, self)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        return toolbox


def custom_automl_mutation(self, individual):
    """Type-aware mutation for unified AutoML chromosome."""
    if random.random() < self.mutation_prob:
        gene_idx = random.randint(0, len(individual) - 1)

        # model_type
        if gene_idx == 0:
            current = int(individual[0])
            choices = [m for m in [0, 1, 2, 3, 4] if m != current]
            individual[0] = random.choice(choices)

        # learning_rate (continuous)
        elif gene_idx == 1:
            lr = float(individual[1]) * random.uniform(0.5, 1.5)
            individual[1] = _clip(lr, AUTO_BOUNDS["learning_rate"][0], AUTO_BOUNDS["learning_rate"][1])

        # batch_size (discrete)
        elif gene_idx == 2:
            individual[2] = random.choice(AUTO_BATCH_SIZES)

        # epochs (+/- 1..5)
        elif gene_idx == 3:
            individual[3] = 100

        # neurons / filters / units (+/-32)
        elif gene_idx == 4:
            val = int(individual[4]) + random.randint(-32, 32)
            individual[4] = int(_clip(val, AUTO_BOUNDS["neurons"][0], AUTO_BOUNDS["neurons"][1]))
        elif gene_idx == 5:
            val = int(individual[5]) + random.randint(-32, 32)
            individual[5] = int(_clip(val, AUTO_BOUNDS["filters"][0], AUTO_BOUNDS["filters"][1]))

        # kernel_size
        elif gene_idx == 6:
            individual[6] = random.choice(AUTO_KERNEL_SIZES)

        elif gene_idx == 7:
            val = int(individual[7]) + random.randint(-32, 32)
            individual[7] = int(_clip(val, AUTO_BOUNDS["units"][0], AUTO_BOUNDS["units"][1]))

        elif gene_idx == 8:  # n_conv_layers
            individual[8] = random.choice([1, 2, 3])

        elif gene_idx == 9:  # pool_size
            individual[9] = random.choice([2, 4])

        elif gene_idx == 10:  # n_dense_layers
            individual[10] = random.choice([1, 2, 3])

        elif gene_idx == 11:  # dense_units
            individual[11] = random.choice([32, 64, 128, 256])

        elif gene_idx == 12:  # dropout_rate
            individual[12] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])

        elif gene_idx == 13:  # optimizer
            individual[13] = random.randint(0, 2)

        elif gene_idx == 14:  # activation
            individual[14] = random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])

    return individual,

def custom_mlp_mutation(self, individual):
        """Mutation personnalisée pour MLP"""
        if random.random() < self.mutation_prob:
            gene_idx = random.randint(0, len(individual) - 1)
            
            if gene_idx == 0:  # n_dense_layers
                individual[gene_idx] = random.randint(1, 5)
            elif gene_idx in range(1, 6):  # dense_units
                individual[gene_idx] = random.choice([32, 64, 128, 256, 512])
            elif gene_idx == 6:  # dropout_rate
                individual[gene_idx] = random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            elif gene_idx == 7:  # learning_rate
                individual[gene_idx] = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01])
            elif gene_idx == 8:  # optimizer
                individual[gene_idx] = random.randint(0, 2)
            elif gene_idx == 9:  # activation
                individual[gene_idx] = random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
            elif gene_idx == 10:  # batch_size
                individual[gene_idx] = random.choice([16, 32, 64, 128])
        
        return individual,


def custom_cnn_mutation(self, individual):
    """Mutation personnalisée pour CNN"""
    if random.random() < self.mutation_prob:
        gene_idx = random.randint(0, len(individual) - 1)
        
        if gene_idx == 0:  # n_conv_layers
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx in range(1, 4):  # conv_filters
            individual[gene_idx] = random.choice([16, 32, 64, 128])
        elif gene_idx in range(4, 7):  # kernel_sizes
            individual[gene_idx] = random.choice([3, 5, 7])
        elif gene_idx in range(7, 10):  # pool_sizes
            individual[gene_idx] = random.choice([2, 4])
        elif gene_idx == 10:  # n_dense_layers
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx in range(11, 16):  # dense_units
            individual[gene_idx] = random.choice([32, 64, 128, 256])
        elif gene_idx == 16:  # dropout_rate
            individual[gene_idx] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        elif gene_idx == 17:  # learning_rate
            individual[gene_idx] = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01])
        elif gene_idx == 18:  # optimizer
            individual[gene_idx] = random.randint(0,2)
        elif gene_idx == 19:  # activation1
            individual[gene_idx] = random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
        elif gene_idx == 20:  # batch_size
            individual[gene_idx] = random.choice([16, 32, 64, 128])
        elif gene_idx == 21:  # n_epochs
            individual[gene_idx] = 100
    
    return individual,

def custom_rnn_mutation(self, individual):
    """Mutation personnalisée pour RNN"""
    if random.random() < self.mutation_prob:
        gene_idx = random.randint(0, len(individual) - 1)
        
        if gene_idx == 0:  # n_rnn_layers
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx == 1:  # rnn_units
            individual[gene_idx] = random.choice([64, 128, 256])
        elif gene_idx == 2:  # n_dense_layers
            individual[gene_idx] = random.choice([1, 2, 3, 4, 5])
        elif gene_idx in range(3, 8):  # dense_units
            individual[gene_idx] = random.choice([64, 128, 256, 512])
        elif gene_idx == 8:  # optimizer
            individual[gene_idx] = random.randint(0, 2)
        elif gene_idx == 9:  # activation
            individual[gene_idx] = random.choice(['relu', 'elu', 'selu', 'tanh'])
        elif gene_idx == 10:  # dropout_rate
            individual[gene_idx] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        elif gene_idx == 11:  # learning_rate
            individual[gene_idx] = random.choice([0.0001, 0.001, 0.005, 0.01, 0.05])
        elif gene_idx == 12:  # batch_size
            individual[gene_idx] = random.choice([16, 32, 64, 128])
        elif gene_idx == 13:  # n_epochs
            individual[gene_idx] = 100
    
    return individual,


def custom_dnn_mutation(self, individual):
    """Mutation for DNN — gene layout:
    [0]   n_hidden_layers  : int   [1-5]
    [1-5] hidden_units     : int   [32,64,128,256,512]
    [6]   dropout_rate     : float [0.0,0.1,0.2,0.3,0.5]
    [7]   learning_rate    : float [0.0001,0.001,0.005,0.01,0.05]
    [8]   optimizer_idx    : int   [0,1,2]
    [9]   activation       : str   ['relu','elu','selu','tanh']
    [10]  batch_size       : int   [16,32,64]
    [11]  n_epochs         : int   [50,100,150]
    """
    if random.random() < self.mutation_prob:
        gene_idx = random.randint(0, len(individual) - 1)

        if gene_idx == 0:                       # n_hidden_layers
            individual[gene_idx] = random.choice([1, 2, 3, 4, 5])
        elif gene_idx in range(1, 6):           # hidden_units
            individual[gene_idx] = random.choice([32, 64, 128, 256, 512])
        elif gene_idx == 6:                     # dropout_rate
            individual[gene_idx] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        elif gene_idx == 7:                     # learning_rate
            individual[gene_idx] = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01])
        elif gene_idx == 8:                     # optimizer_idx
            individual[gene_idx] = random.choice([0, 1, 2])
        elif gene_idx == 9:                     # activation (string)
            individual[gene_idx] = random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
        elif gene_idx == 10:                    # batch_size
            individual[gene_idx] = random.choice([16, 32, 64])
        elif gene_idx == 11:                    # n_epochs
            individual[gene_idx] = 100
    return individual,

def custom_lstm_mutation(self, individual):
    if random.random() < self.mutation_prob:
        gene_idx = random.randint(0, len(individual) - 1)
        
        if gene_idx == 0:  # n_lstm_layers
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx in range(1, 4):  # lstm_units
            individual[gene_idx] = random.choice([32, 64, 128, 256])
        elif gene_idx == 4:  # dropout_rate
            individual[gene_idx] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        elif gene_idx == 5:  # rec_dropout_rate
            individual[gene_idx] = random.choice([0.0, 0.1, 0.2])
        elif gene_idx == 6:  # n_dense_layers
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx in range(7, 10):  # dense_units
            individual[gene_idx] = random.choice([32, 64, 128, 256])
        elif gene_idx == 10:  # learning_rate
            individual[gene_idx] = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01])
        elif gene_idx == 11:  # optimizer_idx
            individual[gene_idx] = random.choice([0, 1, 2])
        elif gene_idx == 12:  # activation
            individual[gene_idx] = random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
        elif gene_idx == 13:  # batch_size
            individual[gene_idx] = random.choice([16, 32, 64])
        elif gene_idx == 14:  # n_epochs
            individual[gene_idx] = 100
    return individual,
# execution_time is not yet defined in this context
    


def run_ga_optimization(self, test='AUTOML', seed=None):
    """Exécution de l'optimisation en séquentiel"""
    _set_global_seed(seed)

    start_time = time.time()
    automl_mode = test in ('AUTOML', 'AUTO', 'ALL', 'UNIFIED')

    if automl_mode:
        # populated during evaluate_individual() calls
        self._automl_best_by_model = {}

    if (test == 'RNN' or test == 'LSTM') and not automl_mode:
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        self.X_val = self.X_val.reshape(self.X_val.shape[0], 1, self.X_val.shape[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])

    # Configuration GA
    toolbox = setup_genetic_algorithm(self, test=test)

    # Population initiale
    population = toolbox.population(n=self.population_size)

    # Statistiques
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of Fame
    hall_of_fame = tools.HallOfFame(1)

    # Évolution (séquentielle)
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=self.crossover_prob,
        mutpb=self.mutation_prob,
        ngen=self.generations,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True
    )

    execution_time = time.time() - start_time

    # Store logbook for per-generation charts
    self.logbook = logbook

    if automl_mode:
        # Unique-model final selection: best tested individual per model type
        per_model = []
        for model_type, entry in getattr(self, '_automl_best_by_model', {}).items():
            ind = entry['individual']
            fit = float(entry['fitness'])
            per_model.append({
                "model_type": int(model_type),
                "individual": ind,
                "fitness": fit,
                "params": decode_automl_individual(ind),
            })

        per_model_sorted = sorted(per_model, key=lambda x: x['fitness'], reverse=True)
        top_k = per_model_sorted[:5]

        self.top_k_individuals = [item["individual"] for item in top_k]
        self.top_k_results = [
            {
                "rank": idx + 1,
                "fitness": item["fitness"],
                "params": item["params"],
            }
            for idx, item in enumerate(top_k)
        ]

        # Best remains rank-1 for compatibility
        if top_k:
            self.best_individual = top_k[0]["individual"]
            self.best_fitness = float(top_k[0]["fitness"])
        else:
            # Fallback (should not happen): keep previous behavior
            sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
            self.best_individual = sorted_population[0]
            self.best_fitness = float(self.best_individual.fitness.values[0])
    else:
        # Meilleur individu (legacy mode)
        self.best_individual = hall_of_fame[0]
        self.best_fitness = self.best_individual.fitness.values[0]

    print(f"⏱️ Temps d'exécution: {execution_time:.2f} secondes")
    metric_name = "validation F1-score" if automl_mode else "recall"
    print(f"🏆 Meilleur fitness ({metric_name}): {self.best_fitness:.4f}")

    if automl_mode and hasattr(self, "top_k_results"):
        print("\nTop Results (best unique model types tested):")
        for item in self.top_k_results:
            p = _automl_relevant_params(item["params"])
            print(
                f"{item['rank']}. {p['model_name']} | "
                f"lr={p['learning_rate']:.5f} | "
                f"batch={p['batch_size']} | "
                f"params={p} | "
                f"fitness={item['fitness']:.4f}"
            )

    return execution_time