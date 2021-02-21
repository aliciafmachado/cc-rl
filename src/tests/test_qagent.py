from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset
from cc_rl.gym_cc.Env import Env
from cc_rl.rl.QAgent import QAgent

dataset = Dataset('emotions')
cc = ClassifierChain()
cc.fit(dataset)
env = Env(cc, dataset.test_x, display="draw")
agent = QAgent(env)
agent.train(10, 2, 10, verbose=True)
print(agent.predict(mode='final_decision'))
