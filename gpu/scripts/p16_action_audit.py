"""Post-mid validation of frozen nav-imitation; no training or selection decisions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from train_chio import minibatches,MidPhase,Policy,recurrent

NAMES=['MOVE','STOP','ATTACK_ENEMY_CREEP','DENY','ATTACK_HERO','ATTACK_TOWER','RAZE_NEAR','RAZE_MID','RAZE_FAR','REQUIEM','USE_FLASK','USE_MANGO','EAT_TREE']

def fractions(matrix,same,previous_known):
    n=int(matrix.sum());rows=[]
    for i,name in enumerate(NAMES):
        actual=int(matrix[i].sum());predicted=int(matrix[:,i].sum());correct=int(matrix[i,i])
        rows.append(dict(action=name,label_share=dict(numerator=actual,denominator=n),recall=dict(numerator=correct,denominator=actual),precision=dict(numerator=correct,denominator=predicted)))
    return dict(frames=n,actions=rows,label_equals_previous_type=dict(numerator=int(same),denominator=n),previous_known=dict(numerator=int(previous_known),denominator=n),confusion_label_rows_prediction_columns=matrix.tolist())

def run(dataset,checkpoint,output):
    started=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    root=Path(dataset);manifest=json.loads((root/'dataset.json').read_text());model=Policy()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);model.eval()
    matrix=np.zeros((13,13),np.int64);same=known=0;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for index,(b,resets,n,_) in enumerate(minibatches([(root,manifest)],'validation',32,20261103,None,[MidPhase(manifest)])):
        labels=b.pop('labels')[:,0];valid=b.pop('loss_valid');previous=b['previous'][:,:13];previous_known=previous.sum(-1)>.5
        with torch.inference_mode():
            tensors={k:torch.from_numpy(v) for k,v in b.items()};h,_=recurrent(model,tensors,n,resets)
            prediction=model.action_head(h).masked_fill(~tensors['action_mask'],-1e9).argmax(-1).numpy()
        np.add.at(matrix,(labels[valid],prediction[valid]),1)
        same+=int(((labels==previous.argmax(-1))&valid&previous_known).sum());known+=int((previous_known&valid).sum())
        (output/'status.json').write_text(json.dumps(dict(batches=index+1,frames=int(matrix.sum()),wall_seconds=time.monotonic()-started)),encoding='utf-8')
    result=fractions(matrix,same,known)
    result.update(dataset=str(root),checkpoint=str(checkpoint),split='validation via train_ggd.split',phase='post-mid with identical MidPhase mask/handoff/reset as training',previous_comparison='Action type (13 classes); previous direction and target are not zeroed.',zero_denominator='Undefined precision/recall when denominator is zero; retained as 0/0.',wall_seconds=time.monotonic()-started)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('checkpoint');p.add_argument('output');run(**vars(p.parse_args()))
