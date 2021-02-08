from src import ClassifierChain
from src import Dataset
from src import Env

dataset = Dataset('birds')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset)
print(env.reset())
steps = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
for i in range(19):
    print(env.step(steps[i]))