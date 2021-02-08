from src import ClassifierChain
from src import Dataset
from src import Env

dataset = Dataset('birds')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc)

env.render()
