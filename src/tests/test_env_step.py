from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset
from cc_rl.gym_cc.Env import Env

dataset = Dataset('emotions')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset.test_x, display='draw')
print(env.reset())
steps = [1, 1, -1, 1, 1, -1]
for i in range(6):
    print(env.step(steps[i]))