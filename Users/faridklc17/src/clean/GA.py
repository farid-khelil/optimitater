
import time
from deap import creator, base, tools, algorithms
import random
import tensorflow as tf
import gc
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from eval import evaluate_individual


def get_mlp_param(toolbox):
    toolbox.register("n_dense_layers", random.randint, 1, 5)
    toolbox.register("dense_units", random.choice, [64, 128, 256, 512])
    toolbox.register("dropout_rate", random.choice, [0.2, 0.3, 0.5])
    toolbox.register("learning_rate", random.choice, [0.001, 0.01, 0.1])
    toolbox.register("optimizer", random.randint, 0, 2)
    toolbox.register("activation", random.randint, 0, 3)
    toolbox.register("batch_size", random.randint, 0, 3)
    toolbox.register("batch_size", random.randint, 0, 3)

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
    toolbox.register("conv_filters", random.choice, [32, 64, 128])
    toolbox.register("kernel_sizes", random.choice, [1, 2, 3])
    toolbox.register("pool_sizes", random.choice, [2, 4, 8])
    toolbox.register("n_dense_layers", random.choice, [1, 2, 3, 4, 5])
    toolbox.register("dense_units", random.choice, [64, 128, 256, 512])
    toolbox.register("dropout_rate", random.choice, [0.0, 0.1, 0.2, 0.3, 0.5])
    toolbox.register("learning_rate", random.choice, [0.0001, 0.001, 0.005, 0.01, 0.05])
    toolbox.register("optimizer", random.randint, 0, 2)
    toolbox.register("activation", random.choice, ['relu', 'elu', 'selu', 'tanh'])
    toolbox.register("batch_size", random.choice,[16, 32, 64, 128])
    toolbox.register("n_epochs", random.choice, [50, 100, 150])

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
    toolbox.register("n_epochs", random.choice, [50, 100, 150])

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
    toolbox.register("learning_rate", random.choice, [0.0001, 0.001, 0.005, 0.01, 0.05])
    toolbox.register("optimizer_idx", random.choice, [0, 1, 2])
    toolbox.register("activation", random.choice, ['relu', 'elu', 'selu', 'tanh'])
    toolbox.register("batch_size", random.choice, [16, 32, 64])
    toolbox.register("n_epochs", random.choice, [50, 100, 150])

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
    toolbox.register("lstm_units", random.choice, [32, 64, 128])
    toolbox.register("dropout_rate", random.choice, [0.0, 0.1, 0.2, 0.3, 0.5])
    toolbox.register("rec_dropout_rate", random.choice, [0.0, 0.1, 0.2])
    toolbox.register("n_dense_layers", random.choice, [1, 2, 3])
    toolbox.register("dense_units", random.choice, [64, 128, 256])
    toolbox.register("learning_rate", random.choice, [0.0001, 0.001, 0.005, 0.01, 0.05])
    toolbox.register("optimizer_idx", random.choice, [0, 1, 2])
    toolbox.register("activation", random.choice, ['relu', 'elu', 'selu', 'tanh'])
    toolbox.register("batch_size", random.choice, [16, 32, 64])
    toolbox.register("n_epochs", random.choice, [50, 100, 150])
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
    
def setup_genetic_algorithm(self, test='MLP'):
        """Configuration de l'AG sans parallélisme"""
        # Création des types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        if test == 'RNN':
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
        if test == 'RNN':
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

def custom_mlp_mutation(self, individual):
        """Mutation personnalisée pour MLP"""
        if random.random() < self.mutation_prob:
            gene_idx = random.randint(0, len(individual) - 1)
            
            if gene_idx == 0:  # n_dense_layers
                individual[gene_idx] = random.randint(1, 5)
            elif gene_idx in range(1, 6):  # dense_units
                individual[gene_idx] = random.choice([64, 128, 256, 512])
            elif gene_idx == 6:  # dropout_rate
                individual[gene_idx] = random.choice([0.2, 0.3, 0.5])
            elif gene_idx == 7:  # learning_rate
                individual[gene_idx] = random.choice([0.001, 0.01, 0.1])
            elif gene_idx == 8:  # optimizer
                individual[gene_idx] = random.randint(0, 2)
            elif gene_idx == 9:  # activation
                individual[gene_idx] = random.randint(0, 3)
            elif gene_idx == 10:  # batch_size
                individual[gene_idx] = random.randint(0, 3)
        
        return individual,


def custom_cnn_mutation(self, individual):
    """Mutation personnalisée pour CNN"""
    if random.random() < self.mutation_prob:
        gene_idx = random.randint(0, len(individual) - 1)
        
        if gene_idx == 0:  # n_conv_layers
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx in range(1, 4):  # conv_filters
            individual[gene_idx] = random.choice([32, 64, 128])
        elif gene_idx in range(4, 7):  # kernel_sizes
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx in range(7, 10):  # pool_sizes
            individual[gene_idx] = random.choice([2, 4, 8])
        elif gene_idx == 10:  # n_dense_layers
            individual[gene_idx] = random.choice([1, 2, 3, 4, 5])
        elif gene_idx in range(11, 16):  # dense_units
            individual[gene_idx] = random.choice([64, 128, 256, 512])
        elif gene_idx == 16:  # dropout_rate
            individual[gene_idx] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        elif gene_idx == 17:  # learning_rate
            individual[gene_idx] = random.choice([0.0001, 0.001, 0.005, 0.01, 0.05])
        elif gene_idx == 18:  # optimizer
            individual[gene_idx] = random.randint(0,2)
        elif gene_idx == 19:  # activation1
            individual[gene_idx] = random.choice(['relu', 'elu', 'selu', 'tanh'])
        elif gene_idx == 20:  # batch_size
            individual[gene_idx] = random.choice([16, 32, 64, 128])
        elif gene_idx == 21:  # n_epochs
            individual[gene_idx] = random.choice([50, 100, 150])
    
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
            individual[gene_idx] = random.choice([50, 100, 150])
    
    return individual,


def custom_dnn_mutation(self, individual):
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
            individual[gene_idx] = random.choice([50, 100, 150])
    return individual,

def custom_lstm_mutation(self, individual):
    if random.random() < self.mutation_prob:
        gene_idx = random.randint(0, len(individual) - 1)
        
        if gene_idx == 0:  # n_lstm_layers
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx in range(1, 4):  # lstm_units
            individual[gene_idx] = random.choice([32, 64, 128])
        elif gene_idx == 4:  # dropout_rate
            individual[gene_idx] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        elif gene_idx == 5:  # rec_dropout_rate
            individual[gene_idx] = random.choice([0.0, 0.1, 0.2])
        elif gene_idx == 6:  # n_dense_layers
            individual[gene_idx] = random.choice([1, 2, 3])
        elif gene_idx in range(7, 10):  # dense_units
            individual[gene_idx] = random.choice([64, 128, 256])
        elif gene_idx == 10:  # learning_rate
            individual[gene_idx] = random.choice([0.0001, 0.001, 0.005, 0.01, 0.05])
        elif gene_idx == 11:  # optimizer_idx
            individual[gene_idx] = random.choice([0, 1, 2])
        elif gene_idx == 12:  # activation
            individual[gene_idx] = random.choice(['relu', 'elu', 'selu', 'tanh'])
        elif gene_idx == 13:  # batch_size
            individual[gene_idx] = random.choice([16, 32, 64])
        elif gene_idx == 14:  # n_epochs
            individual[gene_idx] = random.choice([50, 100, 150])
    return individual,
# execution_time is not yet defined in this context
    


def run_ga_optimization(self, test='MLP'):
        """Exécution de l'optimisation en séquentiel"""
        start_time = time.time()
        if test == 'RNN' or test == 'LSTM':
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
        
        execution_time =  time.time() - start_time
        
        # Meilleur individu
        self.best_individual = hall_of_fame[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        
        print(f"⏱️ Temps d'exécution: {execution_time:.2f} secondes")
        print(f"🏆 Meilleur fitness (recall): {self.best_fitness:.4f}")
        
        return execution_time