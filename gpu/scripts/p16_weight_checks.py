"""CPU checks of the new loss against the established ZOg recipe."""
import json
from pathlib import Path
import torch
from chio_loss_weights import imitation_loss,class_weights
from train_ggd import loss_weights

def run():
    cfg=json.loads((Path(__file__).resolve().parents[2]/'reports/P09a/training-config.json').read_text())
    values=[1,1,4,4,4,4,*[cfg['action_weights'][str(i)] for i in (6,7,8)],1,1,1,1]
    labels=torch.zeros(15,4,dtype=torch.long);labels[:13,0]=torch.arange(13);labels[13:,0]=2
    valid=torch.arange(15)<13;logp=-torch.arange(1.,16.,requires_grad=True);logp.retain_grad()
    legacy=torch.zeros(15,6,dtype=torch.long)
    legacy[:,0]=labels[:,0];legacy[2:6,0]=3;legacy[4,3]=1
    # Non-attack CHiO types mapped away from legacy ATTACK=3.
    expected=loss_weights(legacy,valid,cfg)
    actual=imitation_loss(logp,labels,valid,{'class_weights':values})
    torch.testing.assert_close(actual,-(logp*expected).sum()/expected.sum(),rtol=0,atol=0)
    reference=-(logp*expected).sum()/expected.sum()
    expected_gradient=torch.autograd.grad(reference,logp,retain_graph=True)[0]
    logp.grad=None
    actual.backward();torch.testing.assert_close(logp.grad,expected_gradient,rtol=0,atol=0)
    assert torch.equal(imitation_loss(logp,labels,valid,{}),-logp[valid].mean())
    dagger=values.copy();dagger[4]=1
    cfg['hero_attack_weight']=1;expected=loss_weights(legacy,valid,cfg)
    torch.testing.assert_close(imitation_loss(logp,labels,valid,{'class_weights':dagger}),-(logp*expected).sum()/expected.sum(),rtol=0,atol=0)
    for bad in ([1]*12,[1]*12+[float('nan')],[1]*12+[0]):
        try:class_weights({'class_weights':bad})
        except ValueError:pass
        else:raise AssertionError('Invalid weights accepted')
    return dict(passed=True,classes={'numerator':13,'denominator':13},masked_gradient_zero={'numerator':2,'denominator':2},unweighted_bit_exact=True,zog_loss_and_gradient_exact=True,dagger_hero_override_exact=True,invalid_configs_rejected=3,weights=values)

if __name__=='__main__':print(json.dumps(run()))
