from sklearn.metrics import accuracy_score

from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset

ds = Dataset('emotions')
cc = ClassifierChain()
cc.fit(ds)

y_pred = cc.predict(ds, 'greedy')
print('Greedy exact_match:', accuracy_score(ds.test_y, y_pred))

y_pred = cc.predict(ds, 'qlearning', loss='exact_match', nb_sim=20, nb_paths=5, epochs=5)
print('RL exact_match:', accuracy_score(ds.test_y, y_pred))
