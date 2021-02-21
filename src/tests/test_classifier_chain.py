from sklearn.metrics import hamming_loss

from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.data.Dataset import Dataset

ds = Dataset('emotions')
cc = ClassifierChain()
cc.fit(ds)
y_pred = cc.predict(ds, 'greedy')
print(hamming_loss(ds.test_y, y_pred))
