
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

    toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.n_dense_layers, 
                         toolbox.dense_units, toolbox.dense_units, toolbox.dense_units, toolbox.dense_units, toolbox.dense_units,
                         toolbox.dropout_rate,
                         toolbox.learning_rate,
                         toolbox.optimizer,
                         toolbox.activation,
                         toolbox.batch_size), n=1)



def setup_genetic_algorithm(self):
        """Configuration de l'AG sans parallélisme"""
        # Création des types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        get_mlp_param(toolbox)
        
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Utilisation du map standard (séquentiel)
        toolbox.register("map", map)
        
        toolbox.register("evaluate", evaluate_individual, self)
        toolbox.register("mate", tools.cxTwoPoint)
        # toolbox.evaluate()
        toolbox.register("mutate", custom_mlp_mutation, self)
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

 # execution_time is not yet defined in this context
    


def run_ga_optimization(self):
        """Exécution de l'optimisation en séquentiel"""
        start_time = time.time()
        
        # Configuration GA
        toolbox = setup_genetic_algorithm(self)
        
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