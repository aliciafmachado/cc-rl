from sklearn.metrics import zero_one_loss

from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset

ds = Dataset('emotions')
cc = ClassifierChain()
cc.fit(ds, from_scratch=True)

y_pred = cc.predict(ds, 'random', n=100, loss='exact_match')
print('Random exact_match:', zero_one_loss(ds.test_y, y_pred))

y_pred = cc.predict(ds, 'greedy')
print('Greedy exact_match:', zero_one_loss(ds.test_y, y_pred))

y_pred = cc.predict(ds, 'qlearning', loss='exact_match', nb_sim=20, nb_paths=5, epochs=5)
print('RL exact_match:', zero_one_loss(ds.test_y, y_pred))
