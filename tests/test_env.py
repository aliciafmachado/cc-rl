from src.classifier_chain.ClassifierChain import ClassifierChain
from src.data.Dataset import Dataset
from src.gym_cc.Env import Env

dataset = Dataset('birds')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc)

env.render()
