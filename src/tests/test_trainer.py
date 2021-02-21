from sklearn.metrics import hamming_loss
import time

from cc_rl.classifier_chain.ClassifierChain import ClassifierChain
from cc_rl.classifier_chain.Trainer import Trainer
from cc_rl.data.Dataset import Dataset

ds = Dataset('emotions')
trainer = Trainer(ds)
trainer.train()

t1 = time.time()
cc1 = ClassifierChain()
cc1.fit(ds, from_scratch=True)
t1 = 1000 * (time.time() - t1)

t2 = time.time()
cc2 = ClassifierChain()
cc2.fit(ds)
t2 = 1000 * (time.time() - t2)

print('Hamming loss from scratch ({:.4f} ms fit time): {:.4f}'.format(
    t1, hamming_loss(ds.test_y, cc1.predict(ds, 'beam_search', b=10, loss='hamming'))))
print('Hamming loss trainer ({:.4f} ms fit time): {:.4f}'.format(
    t2, hamming_loss(ds.test_y, cc2.predict(ds, 'beam_search', b=10, loss='hamming'))))
