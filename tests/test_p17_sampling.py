from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'gpu/scripts'))
from p15_model_fixture import batch
from sf_duel.chio_cpu import Session
from sf_duel.chio_model import Policy


def test_sampling_reproducibility_and_conditional_masks():
    torch.set_num_threads(1);model=Policy().cpu().eval()
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
    b,_=batch(model,n=1)
    b['action_mask'][:]=False;b['action_mask'][:,[0,2]]=True
    b['enemy_target_mask'][:]=False;b['enemy_target_mask'][:,[1,3]]=True
    def draw(seed):
        torch.manual_seed(seed);session=Session(model)
        return [session.decide(b,i*6,'test')[0] for i in range(128)]
    first=draw(1701)
    assert first==draw(1701) and first!=draw(1702)
    assert {a['action'] for a in first}=={0,2} # Uniform logits would argmax to0 every time.
    assert len({a['direction'] for a in first if a['action']==0})>1
    assert {a['enemy_target'] for a in first if a['action']==2}=={1,3}
    assert all(a['enemy_target']==0 for a in first if a['action']==0)
    assert all(a['direction']==0 for a in first if a['action']==2)
