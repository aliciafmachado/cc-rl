from sklearn.metrics import hamming_loss

from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.classifier_chain.Trainer import Trainer
from cc_rl.data.Dataset import Dataset

ds = Dataset('emotions')
trainer = Trainer(ds)
trainer.train()

cc1 = ClassifierChain()
cc1.fit(ds, from_scratch=True)

cc2 = ClassifierChain()
cc2.fit(ds)

print('Hamming loss from scratch:',
      hamming_loss(ds.test_y, cc1.predict(ds, 'beam_search', b=10, loss='hamming')))
print('Hamming loss trainer:',
      hamming_loss(ds.test_y, cc2.predict(ds, 'beam_search', b=10, loss='hamming')))
