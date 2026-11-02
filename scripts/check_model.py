"""Check real ROCm LSTM kernels and conditional outputs without training weights."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
import torch
from fixtures import snapshot
from sf_duel.model import Policy
from sf_duel.observations import encode,batch

torch.manual_seed(1)
torch.set_num_threads(2)
model=Policy().to('cuda')
observations=[encode(snapshot()) for _ in range(8)]
b=batch(observations,'cuda')
a,lp,entropy,value,state=model.decide(b)
_,replayed,*_=model.decide(b,actions=a)
torch.testing.assert_close(lp,replayed)
loss=-lp.mean()+.5*value.square().mean()-.01*entropy.mean()
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
torch.cuda.synchronize()
report=dict(passed=True,device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
    recurrent_parameters=sum(p.numel() for p in model.memory.parameters()),batch_size=8,
    gpu_forward_backward=True,conditional_log_probs_replayed=True,
    peak_memory_bytes=torch.cuda.max_memory_allocated(),weights_updated=False)
destination=Path('/mnt/c/AMI/runs/model-check.json');destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
