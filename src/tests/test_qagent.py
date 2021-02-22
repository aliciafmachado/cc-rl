from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset
from cc_rl.gym_cc.Env import Env
from cc_rl.rl.QAgent import QAgent

dataset = Dataset('emotions')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset, display="draw")
agent = QAgent(env)
agent.train(30, 1, 10)
print('no way')
