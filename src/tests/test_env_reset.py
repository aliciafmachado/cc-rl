from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset
from cc_rl.gym_cc.Env import Env

dataset = Dataset('emotions')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset.test_x, display='none')
print(env.reset())
steps = [1, 1, -1, 1, 1]
for i in range(5):
    print(env.step(steps[i]))

# We try to return to a specific label
label_to_return = 3
print("Returning to label " + str(label_to_return))
print(env.reset(label=label_to_return))