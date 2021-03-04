import numpy as np

from cc_rl.rl.MCTSAgent import MCTSAgent
from cc_rl.gym_cc.Env import Env
from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset

def get_greedy(environment):
    next_proba, action_history, _ = environment.reset()
    done = False
    while not done:
        action = int(2 * (next_proba < 0.5) - 1)
        next_proba, action_history, _, final_value, done = environment.step(
            action)

    return (action_history + 1).astype(bool), final_value


def get_random(environment):
    n = 100
    avg = 0
    for i in range(n):
        next_proba, action_history, _ = environment.reset()
        done = False
        while not done:
            action = np.random.randint(0, 2) * 2 - 1
            next_proba, action_history, _, final_value, done = environment.step(
                action)

        avg += final_value

    return avg / n

# Creating environment
sample = 10
dataset = Dataset('emotions')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset.test_x[sample].reshape(1, -1), display=None)

# Initializing agent
agent = MCTSAgent(env)

agent.train(2, 1, 10, verbose=False)

print('Agent prediction: {}, reward: {}'.format(*agent.predict(mode='final_decision',
                                                               return_reward=True)))
print('Greedy prediction: {}, reward: {}'.format(*get_greedy(env)))
print('Average random reward: {}'.format(get_random(env)))

