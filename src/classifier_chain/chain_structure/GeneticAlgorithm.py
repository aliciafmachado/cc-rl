import numpy as np
from typing import List
from src.data.Dataset import Dataset
from src.classifier_chain.ClassifierChain import ClassifierChain
from sklearn.metrics import *

class GeneticAlgorithm:

    def __init__(self, ds: Dataset):
        num_labels = ds.train_y.shape[1]
        individual = ClassifierChain(order=[0, 1, 2, 3])
        individual.fit(ds, optimization=False)
        pred = individual.predict(ds, inference_method="greedy")


    def __start_generation(self):
        # TODO
        pass
    
    def __next_generation(self):
        # TODO
        pass

    def __mutate(self):
        # TODO
        pass
    

class Individual:
    """
    Each Individual correspond to a different label order
    """
    def __init__(self, label_order: List[int]):
        self._fitness = 0
        self.label_order = label_order

    def calculate_fitness(self):
        # TODO
        pass
    
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
    def fittest_individual(self) -> Individual:
        return max(self.individuals, key=lambda individual: individual.fitness)

    def calculate_fitness(self) -> None:
        for individual in self.individuals:
            individual.calculate_fitness()
