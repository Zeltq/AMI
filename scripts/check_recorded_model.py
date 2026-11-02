"""Check random GPU policy against independent batches of real game observations."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import torch
from sf_duel.model import Policy
from sf_duel.observations import encode,batch,target_ref
from sf_duel.spec import Act

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
    run=Path(args.run);torch.set_num_threads(2);torch.manual_seed(1)
    model=Policy().to('cuda').eval();samples=[];counts=Counter();pending=[]
    def verify(rows):
        b=batch(rows,'cuda')
        with torch.inference_mode():
            chosen,logp,_,value,_=model.decide(b)
            _,replayed,*_=model.decide(b,actions=chosen)
            torch.testing.assert_close(logp,replayed)
            assert torch.isfinite(value).all()
            actions={k:v.cpu().tolist() for k,v in chosen.items()}
        for i,e in enumerate(rows):
            a={k:v[i] for k,v in actions.items()};s=e.snapshot
            assert s['legal']['actions'][a['action']] and s['legal']['buys'][a['buy']]
            target=target_ref(e,a['target'],b['creeps'].shape[1],b['trees'].shape[1])
            if a['action']==Act.ATTACK:assert target and target.get('attackable')
            if a['action']==Act.ITEM:
                assert s['legal']['items'][a['item']]
                if a['item']==3:assert target and target.get('tp')
                if a['item']==6:assert target and target.get('type')=='tree'
            counts['observations']+=1;counts['visible_enemy' if s.get('enemy') else 'hidden_enemy']+=1
            counts['creep_rows']+=len(s['creeps']);counts['projectile_rows']+=len(s['projectiles'])
    with gzip.open(run/'snapshots.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            s=json.loads(line)
            if s['seq']%3 or s.get('self',{}).get('health',0)<=0:continue
            pending.append(encode(s))
            if len(pending)==32:verify(pending);pending=[]
    if pending:verify(pending)
    assert counts['visible_enemy']>0 and counts['hidden_enemy']>0 and counts['creep_rows']>0
    report={'passed':True,'device':torch.cuda.get_device_name(),'counts':dict(counts),
        'parameters':sum(p.numel() for p in model.parameters()),'random_weights':True,
        'weights_updated':False,'batch_size':32,'independent_observations':True,
        'conditional_log_probs_replayed':True,'legal_actions_and_exact_targets':True}
    (run/'recorded-model-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
