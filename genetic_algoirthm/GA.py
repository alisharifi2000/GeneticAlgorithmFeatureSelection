import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time

np.random.RandomState(seed=45)
random.seed(45)


class GenticAlgorithmFeatureSelection:
    def __init__(self, features, target, population_size=20, tourn_size=10, mut_rate=0.1, elite_rate=0.4,
                 no_generation=30, fitness_alpha=0.5, method='SVM',
                 method_params={'kernel': 'linear', 'random_state': 42}, k_folds=5, n_job=-1):

        self.population_size = population_size
        self.tourn_size = tourn_size
        self.mut_rate = mut_rate
        self.no_generation = no_generation
        self.features = features
        self.feature_names = np.array(self.features.columns)
        self.target = target.values.reshape(-1)
        self.fitness_alpha = fitness_alpha
        self.no_total_features = self.features.shape[1]
        self.elite_rate = elite_rate
        self.elite_pop = int(self.population_size * self.elite_rate)
        self.method = method
        self.method_params = method_params
        self.population_shape = (self.population_size, self.no_total_features)
        self.selected_featues = None
        self.best_score = None
        self.k_folds = k_folds
        self.n_job = n_job
        self.generation = 0
        self.history = {}

        self.population = self.intital_population()

    def intital_population(self):
        # build initial population
        init_population = [np.ones(shape=self.no_total_features, dtype=bool)]
        population = [self.generate_population() for i in range(self.population_size - 1)]
        population.extend(init_population)
        return population

    def generate_population(self):
        # generate element of population
        pop = np.random.randint(2, size=self.no_total_features, dtype=bool)
        return pop

    def build_model(self):
        # build models by configuration
        if self.method == 'SVM':
            model = SVC(**self.method_params)
            return model

    def fitness_regularization(self, individual):
        # regularization term for fitness score
        count = sum(individual)
        regualrization_score = (1 - (count / self.no_total_features))
        return regualrization_score

    def fitness_method_accuaracy(self, individual):
        # accuracy term for fitness score
        count = sum(individual)
        if count > 0:
            selected_featurs = self.feature_names[individual]
            selected_feature_data = self.features[selected_featurs].values

            model = self.build_model()
            cv = KFold(n_splits=self.k_folds, random_state=42, shuffle=True)

            scores = cross_val_score(model, selected_feature_data, self.target, scoring='accuracy', cv=cv,
                                     n_jobs=self.n_job)
            method_score = np.mean(scores)

        else:
            method_score = 0

        return method_score

    def fitness_function(self, individual):
        method_score = self.fitness_method_accuaracy(individual)
        regualrization_score = self.fitness_regularization(individual)

        score = ((self.fitness_alpha * method_score) + ((1 - self.fitness_alpha) * regualrization_score))
        return score

    def fitness_population(self):
        fitness_scores = [self.fitness_function(individual) for individual in self.population]

        individual_fitness_scores = pd.Series(fitness_scores)
        individual_fitness_scores.sort_values(ascending=False, inplace=True)
        return individual_fitness_scores

    def find_elites(self):
        individual_fitness_scores = self.fitness_population()
        elites_index = individual_fitness_scores.index[:self.elite_pop]
        elites = [self.population[index] for index in elites_index]
        return elites

    def crossover(self, parents1, parents2):
        # one point crossover
        index = random.randint(1, self.no_total_features - 1)

        child1_left_part = np.array(parents1[:index])
        child1_right_part = np.array(parents2[index:])

        child2_left_part = np.array(parents2[:index])
        child2_right_part = np.array(parents1[index:])

        child1 = np.concatenate((child1_left_part, child1_right_part), axis=None)
        child1 = self.maturation(child1)

        child2 = np.concatenate((child2_left_part, child2_right_part), axis=None)
        child2 = self.maturation(child2)

        return [child1, child2]

    def maturation(self, child):
        r = random.random()
        if r <= self.mut_rate:
            child = ~ child

        return child

    def selection(self):
        tourns = random.sample(self.population, self.tourn_size)

        score = [self.fitness_function(individual=torun) for torun in tourns]
        tourns_score = pd.Series(score)
        tourns_score.sort_values(ascending=False, inplace=True)

        index = tourns_score.index[0]
        best_tourn = tourns[index]
        return best_tourn

    def make_child(self):
        parent1 = self.selection()
        parent2 = self.selection()

        return self.crossover(parent1, parent2)

    def make_new_generation(self, verbose):
        s_time = time.time()
        self.generation += 1
        elites = self.find_elites()
        no_make_child = int((self.population_size - self.elite_pop) / 2)
        new_population = [self.make_child() for i in range(no_make_child)]
        new_population = np.array(new_population)
        new_population = new_population.reshape(no_make_child * 2, self.no_total_features)
        new_population = list(new_population)
        new_population.extend(elites)

        if len(new_population) < self.population_size:
            all_features = np.ones(shape=self.no_total_features, dtype=bool)
            new_population.append(all_features)

        self.population = new_population
        self.find_best_result_population(verbose)

        if verbose:
            run_time = time.time() - s_time
            print(f'run_time for generation : {run_time}')

    def find_best_result_population(self, verbose):
        scores = self.fitness_population()
        index = scores.index[0]

        best_score = scores.values[0]
        best_individual = self.population[index]
        selected_featurs = self.feature_names[best_individual]
        self.selected_featues = selected_featurs
        self.best_score = best_score
        self.history[self.generation] = {'best_score': best_score, 'selected_features': selected_featurs}

        if verbose:
            print('---------------------------------------------------------------------')
            print(f'generation :{self.generation} complete !')
            print(f'best score :{self.best_score}')
            print(f'best features:{self.selected_featues}')
            print('---------------------------------------------------------------------')

    def run(self, verbose=True):
        # initial generation
        self.find_best_result_population(verbose)

        # generations
        for generation in range(self.no_generation):
            self.make_new_generation(verbose)
