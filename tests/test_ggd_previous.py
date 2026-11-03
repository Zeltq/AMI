import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from fixtures import snapshot
from sf_duel.model import Policy
from sf_duel.observations import encode,batch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from train_ggd import recurrent
import weights

class GgdPreviousTests(unittest.TestCase):
 def test_zero_previous_matches_zero_input_and_preserves_original(self):
  torch.set_num_threads(2);torch.manual_seed(12);model=Policy(combat_inputs=True).eval()
  b=batch([encode(snapshot(),combat_inputs=True) for _ in range(8)]);b['previous']=torch.randn_like(b['previous']);original=b['previous'].clone()
  zeros={**b,'previous':torch.zeros_like(b['previous'])};resets=np.array([[1,0,1,0],[1,0,0,0]],dtype=np.float32)
  with torch.no_grad():
   actual,keys=recurrent(model,b,2,resets,zero_previous=True);expected,expected_keys=recurrent(model,zeros,2,resets)
   default,_=recurrent(model,b,2,resets);explicit,_=recurrent(model,b,2,resets,zero_previous=False)
  torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(keys,expected_keys,rtol=0,atol=0)
  torch.testing.assert_close(default,explicit,rtol=0,atol=0);torch.testing.assert_close(b['previous'],original,rtol=0,atol=0)
  self.assertGreater(float((actual-default).abs().max()),1e-5)
 def test_export_preserves_ablation_metadata_and_defaults_off(self):
  torch.set_num_threads(2);state=Policy(combat_inputs=True).state_dict()
  with tempfile.TemporaryDirectory() as directory:
   checkpoint=Path(directory)/'input.pt';checkpoint.write_bytes(b'fixture')
   for stored,flag,expected in [(False,False,False),(True,False,True),(False,True,True)]:
    args=['weights.py','--checkpoint',str(checkpoint),'--output',directory]+(['--zero-previous'] if flag else [])
    with patch.object(sys,'argv',args),patch.object(weights.torch,'load',return_value={'state_dict':state,'config':{'zero_previous':stored}}),patch.object(weights,'write_bundle') as write,contextlib.redirect_stdout(io.StringIO()):weights.main()
    self.assertEqual(write.call_args.kwargs['zero_previous'],expected)

if __name__=='__main__':unittest.main()
