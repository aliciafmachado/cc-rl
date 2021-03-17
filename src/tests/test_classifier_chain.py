from sklearn.metrics import zero_one_loss, hamming_loss

from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset

ds = Dataset('emotions')
cc = ClassifierChain()
cc.fit(ds)

y_pred, reward = cc.predict(ds, 'random', loss='exact_match', n=100, return_reward=True)
print('Random exact_match:', zero_one_loss(ds.test_y, y_pred))
print('Random reward:', reward)

y_pred, reward = cc.predict(ds, 'greedy', loss='exact_match', return_reward=True)
print('Greedy exact_match:', zero_one_loss(ds.test_y, y_pred))
print('Greedy reward:', reward)

y_pred, reward = cc.predict(ds, 'exhaustive_search', loss='exact_match', return_reward=True)
print('exhaustive_search exact_match:', zero_one_loss(ds.test_y, y_pred))
print('exhaustive_search reward:', reward)

y_pred, reward = cc.predict(ds, 'qlearning', loss='exact_match', nb_sim=100, nb_paths=3,
                            epochs=10, return_reward=True)
print('RL exact_match:', zero_one_loss(ds.test_y, y_pred))
print('RL reward:', reward)

y_pred, reward = cc.predict(ds, 'mcts', loss='exact_match', nb_sim=60, nb_paths=1,
                            epochs=10, mcts_passes=3, return_reward=True)
print('RL exact_match:', zero_one_loss(ds.test_y, y_pred))
print('RL reward:', reward)
