from cc_rl.data.Dataset import Dataset
from cc_rl.classifier_chain.chain_structure.GeneticAlgorithm import GeneticAlgorithm

ds = Dataset('emotions')
ga = GeneticAlgorithm(ds, 30, 100)
ga.run(3)
