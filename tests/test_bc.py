import unittest
import numpy as np
import torch
from fixtures import snapshot
from sf_duel.observations import encode,batch,numpy_batch
from sf_duel.model import Policy
from sf_duel.bc_data import sequence_batch

class SequenceTests(unittest.TestCase):
 def test_sequence_matches_stepwise_recurrent_policy(self):
  torch.manual_seed(4);model=Policy().eval();n,t=2,5
  b=batch([encode(snapshot()) for _ in range(n*t)])
  with torch.no_grad():
   seq=model.sequence_decide(b,n,t,deterministic=True);state=None;logs=[]
   for step in range(t):
    indices=torch.arange(n)*t+step
    result=model.decide({k:v[indices] for k,v in b.items()},state,deterministic=True);state=result[-1];logs.append(result[1])
   torch.testing.assert_close(seq[1].reshape(n,t),torch.stack(logs,1),rtol=1e-5,atol=1e-5)
   for actual,expected in zip(seq[-1],state):torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
 def test_burnin_padding_and_targets(self):
  e=encode(snapshot());b=numpy_batch([e.arrays]*100)
  b.update(labels=np.tile(np.array([0,0,0,0,64,64]),(100,1)),loss_valid=np.ones(100,bool))
  x,y,mask=sequence_batch([b],[(0,0),(0,64)],80,16)
  self.assertEqual(mask[:80].sum(),80)
  self.assertEqual(mask[80:].sum(),20)
  self.assertEqual(x['heroes'].shape[0],160)
  self.assertTrue(np.isfinite(x['heroes']).all())
