import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import time

np.random.RandomState(seed=45)
random.seed(45)

#TODO: add logger
#TODO: add XGboost, Lightgbm model
#TODO: build requirement.txt


class FeatureSelection:
    def __init__(self, features, target, population_size=20, tourn_size=10, mut_rate=0.1, elite_rate=0.4,
                 no_generation=30, fitness_alpha=0.5, method='SVM',
                 method_params={}, k_folds=5, scoring='f1_score', n_job=-1):
        '''

        :param features: pandas dataframe of all features
        :param target: pandas dataframe of classification labels
        :param population_size: size of population (default 20)
        :param tourn_size: size of selected choromosom for tournoment as paraents (default 10)
        :param mut_rate: rate of maturation (default 0.1)
        :param elite_rate: rate of elite transfer to next generation (default 0.4)
        :param no_generation: number of generation (default 30)
        :param fitness_alpha: parameter for trade of between regularization term and accuracy term (default 0.5)
        :param method: name of classification method (default SVM)
        :param method_params: parameters of classification method
        :param k_folds: number of k-fold cross validtion (default 5)
        :param n_job: number of jobs
        '''

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
        self.scoring = scoring
        self.n_job = n_job
        self.generation = 0
        self.history = {}

        self.population = self.intital_population()

    def intital_population(self):
        """
        build initial population
        :return: list of numpy array each of element represent as chromosome
        """

        init_population = [np.ones(shape=self.no_total_features, dtype=bool)]
        population = [self.generate_population() for i in range(self.population_size - 1)]
        population.extend(init_population)
        return population

    def generate_population(self):
        """
        generate chromosome of population
        :return: chromosome represent existance of feature in featues list
        """

        pop = np.random.randint(2, size=self.no_total_features, dtype=bool)
        return pop

    def build_model(self):
        """
        build models by configuration
        :return: sklearn models by configuration
        """
        try:
            if self.method == 'SVM':
                model = SVC(**self.method_params)
                return model

            elif self.method == 'DesicionTree':
                model = DecisionTreeClassifier(**self.method_params)
                return model

            elif self.method == 'LogesticRegression':
                model = LogisticRegression(**self.method_params)
                return model

            elif self.method == 'KNN':
                model = KNeighborsClassifier(**self.method_params)
                return model

            elif self.method == 'NavieBayes':
                model = GaussianNB(**self.method_params)
                return model

            elif self.method == 'RandomForest':
                model = RandomForestClassifier(self.method_params)
                return model

        except Exception as e:
            print(e)

    def fitness_regularization(self, individual):
        """
        regularization term for fitness score
        :param individual: one chromosome
        :return: regularization score. percentage of feature not exist in feature list
        """

        count = sum(individual)
        regularization_score = (1 - (count / self.no_total_features))
        return regularization_score

    def make_score(self):
        if self.scoring == 'accuracy':
            return make_scorer(accuracy_score)
        elif self.scoring == 'precision':
            return make_scorer(precision_score),
        elif self.scoring == 'recall':
            return make_scorer(recall_score)
        elif self.scoring == 'f1_score':
            return make_scorer(f1_score)

    def fitness_method_accuaracy(self, individual):
        """
        accuracy term for fitness score
        :param individual: one chromosome
        :return: accuracy of model for chromosome
        """
        count = sum(individual)
        if count > 0:
            selected_featurs = self.feature_names[individual]
            selected_feature_data = self.features[selected_featurs].values

            model = self.build_model()
            cv = KFold(n_splits=self.k_folds, random_state=42, shuffle=True)

            scores = cross_val_score(model, selected_feature_data, self.target, scoring=self.make_score(),
                                     cv=cv, n_jobs=self.n_job)
            method_score = np.mean(scores)

        else:
            method_score = 0

        return method_score

    def fitness_function(self, individual):
        """
        fitness function for chromosome
        :param individual: one chromosome
        :return: fitness score of chromosome
        """
        method_score = self.fitness_method_accuaracy(individual)
        regularization_score = self.fitness_regularization(individual)

        score = ((self.fitness_alpha * method_score) + ((1 - self.fitness_alpha) * regularization_score))
        return score

    def fitness_population(self):
        """
        find fitness score for chromosomes in population
        :return: list of fitness score of population
        """
        fitness_scores = [self.fitness_function(individual) for individual in self.population]

        individual_fitness_scores = pd.Series(fitness_scores)
        individual_fitness_scores.sort_values(ascending=False, inplace=True)
        return individual_fitness_scores

    def find_elites(self):
        """
        find elite members
        :return: list of elite chromosomes
        """
        individual_fitness_scores = self.fitness_population()
        elites_index = individual_fitness_scores.index[:self.elite_pop]
        elites = [self.population[index] for index in elites_index]
        return elites

    def crossover(self, parents1, parents2):
        """
        one point crossover
        :param parents1: chromosome as paraent1
        :param parents2: chromosome as paraent2
        :return: two children by one point crossover
        """
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
        """
        maturation one chromosome by mut_rate
        :param child: chromosome
        :return: maturation one chromosome
        """
        r = random.random()
        if r <= self.mut_rate:
            child = ~ child
        return child

    def selection(self):
        """
        find best chromosome in tournment for being parent
        :return: best chromosome in tournment for being parent
        """
        tourns = random.sample(self.population, self.tourn_size)

        score = [self.fitness_function(individual=torun) for torun in tourns]
        tourns_score = pd.Series(score)
        tourns_score.sort_values(ascending=False, inplace=True)

        index = tourns_score.index[0]
        best_tourn = tourns[index]
        return best_tourn

    def make_child(self):
        """
        build two child from two parants by one point crossover
        :return: list of two child
        """
        parent1 = self.selection()
        parent2 = self.selection()

        return self.crossover(parent1, parent2)

    def make_new_generation(self, verbose):
        """
        build new generation and find best feature set by generation
        :param verbose: status that show result of each generation (default True)
        """
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
        """
        find best result of generation and set feature set
        :param verbose:
        """
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
        """
        run process
        :param verbose:
        :return:
        """
        # initial generation
        self.find_best_result_population(verbose)

        # generations
        for generation in range(self.no_generation):
            self.make_new_generation(verbose)
