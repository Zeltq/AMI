"""Explicit CHiO class weights, normalized as in ZOg train_ggd.loss_weights."""
import math
import torch

def class_weights(config):
    values=config.get('class_weights',[1.]*13)
    if len(values)!=13 or any(not math.isfinite(float(v)) or float(v)<1 for v in values):
        raise ValueError('CHiO class_weights requires 13 finite weights >= 1')
    return [float(v) for v in values]

def imitation_loss(logp,labels,valid,config):
    # Preserve the old arithmetic exactly for unweighted training.
    values=class_weights(config)
    if all(v==1 for v in values):return -logp[valid].mean()
    weights=torch.as_tensor(values,device=logp.device,dtype=logp.dtype)[labels[:,0]][valid]
    return -(logp[valid]*weights).sum()/weights.sum().clamp_min(1)
