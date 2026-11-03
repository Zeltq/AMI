import importlib.util,tempfile,unittest,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpu/scripts'))
from train_ggd import load_resume
from sf_duel.model import Policy

class GgdResumeTests(unittest.TestCase):
 def test_weights_optimizer_and_next_epoch_survive_cpu_restore(self):
  torch.set_num_threads(2);torch.manual_seed(7);model=Policy(combat_inputs=True)
  optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
  first=torch.randn(3,256);model.action_head(first).square().sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
  config=dict(seed=7,batch_sequences=8,learning_rate=1e-4,gradient_norm=1.,attack_weight=4.,epochs=5,observation_profile='ggd-v2',dataset_sha256='dataset')
  saved=dict(spec_version=2,config=config,epoch=2,history=[dict(epoch=i,train={'frames':128},validation={'frames':128,'nll':2.}) for i in (1,2)],state_dict=model.state_dict(),optimizer=optimizer.state_dict(),optimizer_steps=1)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'checkpoint.pt';torch.save(saved,path);checkpoint=load_resume(path,config,'dataset')
   clone=Policy(combat_inputs=True);clone.load_state_dict(checkpoint['state_dict']);other=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-5);other.load_state_dict(checkpoint['optimizer'])
   second=torch.randn(3,256)
   for network,opt in [(model,optimizer),(clone,other)]:
    opt.zero_grad(set_to_none=True);network.action_head(second).square().sum().backward();opt.step()
   torch.testing.assert_close(model.action_head.weight,clone.action_head.weight,rtol=0,atol=0)
   self.assertEqual(int(other.state[clone.action_head.weight]['step']),2)
   self.assertEqual(checkpoint['epoch']+1,3)
   with self.assertRaisesRegex(ValueError,'dataset'):load_resume(path,config,'different')
   with self.assertRaisesRegex(ValueError,'seed'):load_resume(path,dict(config,seed=8),'dataset')
   with self.assertRaisesRegex(ValueError,'zero_previous'):load_resume(path,dict(config,zero_previous=True),'dataset')

if __name__=='__main__':unittest.main()
