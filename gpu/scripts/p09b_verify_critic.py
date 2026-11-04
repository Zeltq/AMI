"""CPU integration check on real v4 sequences before allocating the GPU critic job."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from p09b_critic import actor_hash,ev_stats,summarize
from train_ggd import recurrent
from sf_duel.model import Policy
from sf_duel.observations import packed_tensors
from sf_duel.timing_observations import ablate
from ggd_dataset import read_block,decode

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('dataset');p.add_argument('targets');p.add_argument('output');p.add_argument('--block',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261105)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=Policy(combat_inputs=True,timing_inputs=True,enemy_inputs=True);model.load_state_dict(saved['state_dict']);model.eval()
    before=actor_hash(model)
    for param in model.parameters():param.requires_grad_(False)
    root=Path(a.dataset);manifest=json.loads((root/'dataset.json').read_text());entry=next(e for e in manifest['files'] if e['file'].startswith(f'block-{a.block:06}-shard-000.'))
    header,obs,signals=read_block(root/entry['file'],entry['sha256'])
    ids=[i for i,actor in enumerate(header['actors']) if actor//2%8 in (0,6)]
    arrays=decode(header,obs[ids],signals[ids]);valid=arrays.pop('loss_valid').astype(bool);arrays.pop('labels')
    arrays=ablate(arrays,True,False)
    with torch.no_grad():hidden,_=recurrent(model,packed_tensors(arrays,'cpu'),len(ids),signals[ids,:,10])
    target_root=Path(a.targets);targets=json.loads((target_root/'targets.json').read_text())
    reward=next(r for r in targets['files'] if r['demonstration']==entry['file'])
    with np.load(target_root/reward['file']) as z:
        assert z['actors'].tolist()==header['actors'];y=z['returns'][ids].reshape(-1)[valid];post=z['post0'][ids].reshape(-1)[valid]
    x=hidden[valid];cached=x.half().float()
    assert len(y)>0 and np.isfinite(y).all()
    with torch.no_grad():
        prediction=model.value_head(x).squeeze(-1)
        quantized=model.value_head(cached).squeeze(-1)
        error=float((prediction-quantized).abs().max());assert error<1e-3
    initial={k:v.clone() for k,v in model.value_head.state_dict().items()}
    for param in model.value_head.parameters():param.requires_grad_(True)
    optimizer=torch.optim.Adam(model.value_head.parameters(),lr=3e-4)
    loss=(model.value_head(cached).squeeze(-1)-torch.as_tensor(y)).square().mean()
    loss.backward();torch.nn.utils.clip_grad_norm_(model.value_head.parameters(),.5);optimizer.step()
    assert actor_hash(model)==before
    changed=sum(not torch.equal(v,initial[k]) for k,v in model.value_head.state_dict().items());assert changed==4
    result=dict(passed=True,device='CPU',checkpoint=a.checkpoint,demonstration=entry['file'],actors=[header['actors'][i] for i in ids],frames=len(y),post0_frames=int(post.sum()),pregame_frames=int((~post).sum()),actor_sha256_before=before,actor_sha256_after=actor_hash(model),value_tensors_changed=changed,value_tensors=4,fp16_cache_max_value_difference=error,target_min=float(y.min()),target_max=float(y.max()),loss=float(loss.detach()),scope='one real recurrent block, both train and validation actor identities; one value update; not a full critic training or GPU compatibility check')
    Path(a.output).write_text(json.dumps(result,indent=2),newline='\n');print(json.dumps(result))

if __name__=='__main__':main()
