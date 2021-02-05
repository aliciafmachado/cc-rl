import numpy as np
from typing import List
from src.data.Dataset import Dataset
# from src.classifier_chain.ClassifierChain import ClassifierChain
import random

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
    
    def run(self, n: int):
        """
        run the Genetic Algorithm for n generations
        
        Args:
            n (int): number of generations

        Returns:
            order (List[int]): optimal order of labels
        """
        self.__start_generation()
        self.__calculate_fitness()
        for i in range(n):
            self.__next_generation()
            self.__calculate_fitness()
            self.__show(i)

        return self.population.fittest_individual()


    def __calculate_fitness(self):
        for individual in self.population.individuals:
            individual.cc.fit(self.ds, optimization=False)
            individual.calculate_fitness(self.ds)

    def __start_generation(self):
        for i in range(self.k):
            order = np.arange(self.num_labels)
            np.random.shuffle(order)
            individual = Individual(order)
            self.population.individuals.append(individual)

    def __next_generation(self):
        players = random.sample(self.population.individuals, self.k)
        winners = self.population.fittest_individual(players)
        new_population = []
        while len(new_population) < self.num_individuals and len(self.population.individuals) > 0:
            # select two individuals and remove them
            p1, p2 = random.sample(self.population.individuals, 2)
            self.population.individuals.remove(p1)
            self.population.individuals.remove(p2)


        new_population = random.sample(new_population, self.num_individuals-2)
        new_population.extend(winners)
        random.shuffle(new_population)
        self.population.individuals = new_population


    def __mutate(self, p1, p2):
        # TODO
        pass
    
    def __show(self, generation):
        # TODO
        pass


class Individual:
    """
    Each Individual correspond to a different label order
    """
    def __init__(self, label_order: List[int]):
        self._fitness = 0
        self.label_order = label_order
        self.cc = ClassifierChain(order=label_order)

    def calculate_fitness(self, ds):
        acc = self.cc.accuracy(ds)
        hl = self.cc.hamming_loss(ds)
        em = self.cc.exact_match(ds)
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
    
    @property
    def fittest_individual(self, players) -> Individual:
        # return two most fittest individuals (elitism selection)
        return sorted(players, key=lambda individual: individual.fitness)[-2:]

    def calculate_fitness(self) -> None:
        for individual in self.individuals:
            individual.calculate_fitness()
