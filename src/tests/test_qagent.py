import numpy as np

from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset
from cc_rl.gym_cc.Env import Env
from cc_rl.rl.QAgent import QAgent


def get_greedy(environment):
    next_proba, action_history, proba_history = environment.reset()
    done = False
    while not done:
        action = np.argmax(next_proba)
        next_proba, action_history, proba_history, final_value, done = environment.step(
            action)

    return action_history.astype(bool), final_value


dataset = Dataset('emotions')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset.test_x, display="draw")
agent = QAgent(env)
agent.train(10, 2, 10, verbose=True)
print('Agent prediction: {}, reward: {}'.format(*agent.predict(mode='final_decision',
                                                               return_reward=True)))
print('Greedy prediction: {}, reward: {}'.format(*get_greedy(env)))
