import torch

from cc_rl.rl.MCTSModel import MCTSModel
from cc_rl.rl.MCTSAgent import MCTSAgent
from cc_rl.gym_cc.Env import Env
from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset

# Creating environment
sample = 15
dataset = Dataset('emotions')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset.test_x[sample].reshape(1, -1), display="none")

# Initializing agent
agent = MCTSAgent(env)

agent.experience_environment_once()