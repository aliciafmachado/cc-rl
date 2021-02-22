import torch

from cc_rl.rl.MCTSModel import MCTSModel
from cc_rl.gym_cc.Env import Env
from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset

# Creating environment
sample = 15
dataset = Dataset('emotions')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset.test_x[sample].reshape(1, -1), display="none")

# Creating model
model = MCTSModel(env.classifier_chain.n_labels + 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Passing some inpouts through the model
actions_history = torch.tensor([[1, 1, -1, 0, 0, 0], [1, 1, -1, 1, 0, 0]])
probas_history = torch.tensor([[0.5, 0.3, 0.2, 0, 0, 0], [0.5, 0.3, 0.2, 0.9, 0, 0]])
next_probas = torch.tensor([0.9, 0.5]).reshape(-1, 1)

print(model(actions_history, probas_history, next_probas))
