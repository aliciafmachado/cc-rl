import numpy as np
from typing import List, Optional
import random
import itertools
from sklearn.metrics import accuracy_score, hamming_loss
import json

from cc_rl.data.Dataset import Dataset
from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.utils.multilabel_accuracy import multilabel_accuracy


class GeneticAlgorithm:

    def __init__(self, ds: Dataset, k: int, num_individuals: int):
        """
        Args:
            k : number of players selected to the tournament
            num_individuals: number of players
        """
        self.ds = ds
        self.population = Population([])
        self.num_labels = ds.train_y.shape[1]
        self.k = k
        self.num_individuals = num_individuals
    
    def run(self, n: int, verbose: bool = False):
        """
        run the Genetic Algorithm for n generations
        
        Args:
            n (int): number of generations
            verbose (bool): print progress
            path (str): path to save the best label order into a json. If None, it will not save 
        Returns:
            order (List[int]): optimal order of labels
        """
        self.__start_generation()
        self.__calculate_fitness()
        for i in range(n):
            self.__next_generation()
            self.__calculate_fitness()
            if verbose:
                self.__show(i)
        return self.population.fittest_two_individual(self.population.individuals)

    def save(self, path):
        """
        save the Genetic Algorithm into a json
        
        Args:
            path (str): path to save the best label order into a json 
        Returns:
            order (List[int]): optimal order of labels
        """
        individual = self.population.fittest_individual()
        order = [int(l) for l in individual.label_order]
        fitness = individual.fitness
        data = {'name': self.ds.name,
                'num_labels': len(order),
                'order': order,
                'fitness': fitness
            }
        with open(path, 'w') as f:
            json.dump(data, f)

    def __calculate_fitness(self):
        for individual in self.population.individuals:
            individual.cc.fit(self.ds, from_scratch=True)
            individual.calculate_fitness(self.ds)

    def __start_generation(self):
        for _ in range(self.k):
            individual = self.__new_individual()
            self.population.individuals.append(individual)

    def __next_generation(self):
        players = random.sample(self.population.individuals, self.k)
        winners = self.population.fittest_two_individual(players)
        new_population = []
        while len(players) > 1:       
            # select two individuals and remove them
            p1, p2 = random.sample(players, 2)
            players.remove(p1)
            players.remove(p2)

            # crossover
            individual = self.__crossover(p1, p2)
            new_population.append(individual)

        # mutate one child
        self.__mutate(random.choice(new_population))

        # elitism selection
        new_population.extend(winners)
        
        # complete new generation with random individuals
        while len(new_population) < self.num_individuals:
            individual = self.__new_individual()
            new_population.append(individual)

        random.shuffle(new_population)
        self.population.individuals = new_population


    def __crossover(self, p1, p2):
        donner = list(p1.label_order.copy())
        receptor = list(p2.label_order.copy())

        # select sub-chain from p1 with uniform prob
        sub_chain = random.choice(list(itertools.combinations(range(self.num_labels + 1), r=2)))
        sub_chain = list(sub_chain)
        sub_chain[1] -= 1

        # remove sub_chain itens from p2
        for i in range(sub_chain[0], sub_chain[1] + 1):
            receptor.remove(donner[i])

        if len(receptor) == 0:
            new_order = donner
            individual = Individual(new_order)
            return individual

        # extend sub_chain
        j = np.random.randint(len(receptor))
        new_order = receptor[:j]
        new_order.extend(donner[sub_chain[0]:sub_chain[1]+1])
        new_order.extend(receptor[j:])
        individual = Individual(new_order)
        return individual

    def __mutate(self, p):
        idx = range(self.num_labels)
        i1, i2 = random.sample(idx, 2)
        p.label_order[i1], p.label_order[i2] = p.label_order[i2], p.label_order[i1]

    def __new_individual(self):
        order = np.arange(self.num_labels)
        np.random.shuffle(order)
        individual = Individual(order)
        return individual   

    def __show(self, generation):
        p1, p2 = self.population.fittest_two_individual(self.population.individuals)
        print("Generation {}".format(generation))
        print("Number of individuals = {}".format(len(self.population.individuals)))
        print("Elitism selection")
        print("Individual 1 = {}, score = {:.2f}".format(p1.label_order, p1.fitness))
        print("Individual 2 = {}, score = {:.2f}".format(p2.label_order, p2.fitness))
        print()


class Individual:
    """
    Each Individual correspond to a different label order
    """
    def __init__(self, label_order: List[int]):
        self._fitness = 0
        self.label_order = np.array(label_order)
        self.cc = ClassifierChain(order=label_order)

    def calculate_fitness(self, ds):
        y_pred = self.cc.predict(ds, 'greedy')
        acc = multilabel_accuracy(ds.test_y, y_pred)
        hl = hamming_loss(ds.test_y, y_pred)
        em = accuracy_score(ds.test_y, y_pred)
        self._fitness = (acc + em + (1 - hl)) / 3

    @property
    def fitness(self):
        return self._fitness


class Population:
    """
    Population of Individuals
    """
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @property
    def num_individuals(self) -> int:
        return len(self.individuals)
    
    def fittest_two_individual(self, players) -> Individual:
        # return two most fittest individuals (elitism selection)
        return sorted(players, key=lambda individual: individual.fitness)[-2:]

    def fittest_individual(self) -> Individual:
        return max(self.individuals, key=lambda individual: individual.fitness)

    def calculate_fitness(self) -> None:
        for individual in self.individuals:
            individual.calculate_fitness()