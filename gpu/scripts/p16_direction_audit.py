"""CPU-only direction audit on frozen imitation weights and recorded public inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from ggd_dataset import read_block,decode,layout
from train_ggd import recurrent,split
from p15_model_fixture import Policy

def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
def new():return dict(frames=0,correct=0,student_move=0,joint_correct=0,confusion=np.zeros((36,36),np.int64),student_executed_frames=0,student_executed_correct=0,delta_histogram=np.zeros(36,np.int64),toward_mid_histogram=np.zeros(36,np.int64))
def finish(x):
    matrix=x.pop('confusion');x['confusion_teacher_rows_student_columns']=matrix.tolist();n=x['frames']
    offsets=[sum(matrix[t,(t+k)%36] for t in range(36)) for k in range(36)];mirrors=[sum(matrix[t,(-t+k)%36] for t in range(36)) for k in range(36)]
    x.update(best_offset_bin=int(np.argmax(offsets)),best_offset_matches=int(max(offsets)),best_mirror_offset_bin=int(np.argmax(mirrors)),best_mirror_matches=int(max(mirrors)),denominator=n)
    for k,v in list(x.items()):
        if isinstance(v,np.ndarray):x[k]=v.tolist()
    return x

def predict(model,header,obs,sig):
    b=decode(header,obs,sig);labels=b.pop('labels');valid=b.pop('loss_valid');b={k:torch.from_numpy(v) for k,v in b.items()}
    with torch.inference_mode():
        h,keys=recurrent(model,b,len(obs),sig[:,:,10]);kind=model.action_head(h).masked_fill(~b['action_mask'],-1e9).argmax(-1)
        forced=torch.zeros(len(h),dtype=torch.long);direction=model.direction_head(torch.cat((h,model.action_embed(forced)),-1)).argmax(-1)
    return kind.numpy().reshape(sig.shape[:2]),direction.numpy().reshape(sig.shape[:2])

def add(out,side,mask,teacher,student,kind,student_used,xy):
    x=out['radiant' if side==0 else 'dire'];t=teacher[mask].astype(int);s=student[mask].astype(int);k=kind[mask];used=student_used[mask]
    x['frames']+=len(t);x['correct']+=int((t==s).sum());x['student_move']+=int((k==0).sum());x['joint_correct']+=int(((k==0)&(t==s)).sum());x['student_executed_frames']+=int(used.sum());x['student_executed_correct']+=int(((s==t)&used).sum())
    np.add.at(x['confusion'],(t,s),1);delta=(s-t+18)%36-18;np.add.at(x['delta_histogram'],delta+18,1)
    points=xy[mask];mid=np.floor(np.arctan2(-points[:,1],-points[:,0])/np.deg2rad(10)+.5).astype(int)%36;np.add.at(x['toward_mid_histogram'],(s-mid+18)%36,1)

def run(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);model=Policy();model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False)['state_dict']);model.eval();started=time.monotonic()
    groups={'teacher_validation_premid':{s:new() for s in ['radiant','dire']},'dagger_learner_stuck':{s:new() for s in ['radiant','dire']}}
    root=a.teacher;m=json.loads((root/'dataset.json').read_text());arrivals={g['game']*2+side:s['mid_arrival_elapsed_seconds'] for g in m['game_metrics'] for side,s in enumerate(g['sides'])};finished=set();files=0;actor_steps={}
    for entry in m['files']:
        ids=[i for i,actor in enumerate(entry['actors']) if split(actor)=='validation' and actor not in finished]
        if not ids:continue
        h,o,s=read_block(root/entry['file'],entry['sha256']);b=decode(h,o[ids],s[ids]);times=b['global_features'][:,0].reshape(len(ids),128)*600;xy=b['heroes'][:,0,:2].reshape(len(ids),128,2)*8192
        masks=[]
        for row,i in enumerate(ids):
            actor=h['actors'][i];arrival=arrivals[actor];elapsed=(actor_steps.get(actor,0)+np.arange(128))*.2;actor_steps[actor]=actor_steps.get(actor,0)+128;pre=np.ones(128,bool) if arrival is None else elapsed<arrival-1e-4
            masks.append(pre&(s[i,:,11]>.5)&(s[i,:,0]==0))
            if arrival is not None and elapsed[-1]>=arrival:finished.add(actor)
        masks=np.asarray(masks);maximum=max((int(np.flatnonzero(mask)[-1])+1 for mask in masks if mask.any()),default=0)
        if maximum:
            for begin in range(0,len(ids),8):
                selected=ids[begin:begin+8];kind,direction=predict(model,h,o[selected,:maximum],s[selected,:maximum])
                for j,i in enumerate(selected):add(groups['teacher_validation_premid'],h['actors'][i]%2,masks[begin+j,:maximum],s[i,:maximum,1],direction[j],kind[j],np.zeros(maximum,bool),xy[begin+j,:maximum])
        files+=1
        save(a.output/'status.json',dict(stage='teacher-validation',files=files,actors_finished=len(finished),wall_seconds=time.monotonic()-started))
    save(a.output/'teacher-validation.json',{side:finish(value) for side,value in groups['teacher_validation_premid'].items()})
    root=a.dagger;m=json.loads((root/'dataset.json').read_text());positions={};selections=[];groups_layout,_=layout(5);total_learner=0;stuck=0
    for entry in m['files']:
        ids=[i for i,actor in enumerate(entry['actors']) if actor%2==(actor//2)%2]
        if not ids:continue
        h,o,s=read_block(root/entry['file'],entry['sha256']);inverse={col:i for i,col in enumerate(h['columns'])}
        hero=groups_layout['heroes'][0];previous=groups_layout['chio_previous'][0];owncast=groups_layout['hero_chio'][0]+20
        xy=o[ids][:,:,[inverse[hero],inverse[hero+1]]].astype(np.float32)*8192;speed=o[ids,:,inverse[hero+15]].astype(np.float32)*600
        moving=o[ids,:,inverse[previous]]>.5;casting=o[ids,:,inverse[owncast]]>.5;masks=np.zeros((len(ids),128),bool)
        for row,i in enumerate(ids):
            actor=h['actors'][i];last,slow=positions.get(actor,(None,0))
            for t in range(128):
                if s[i,t,11]<=.5:continue
                total_learner+=1;point=xy[row,t];distance=np.linalg.norm(point-last) if last is not None else 1e9
                slow=slow+1 if last is not None and moving[row,t] and not casting[row,t] and distance<.5*speed[row,t]*.2 else 0
                masks[row,t]=slow>=6 and s[i,t,0]==0
                last=point.copy()
            positions[actor]=(last,slow)
        if masks.any():selections.append((entry,ids,masks));stuck+=int(masks.sum())
    save(a.output/'status.json',dict(stage='dagger-inference',selected_files=len(selections),stuck_frames=stuck,total_learner_frames=total_learner,wall_seconds=time.monotonic()-started))
    for number,(entry,ids,masks) in enumerate(selections):
        h,o,s=read_block(root/entry['file'],entry['sha256']);selected=[(i,row) for row,i in enumerate(ids) if masks[row].any()]
        for begin in range(0,len(selected),8):
            chunk=selected[begin:begin+8];rows=[i for i,_ in chunk];kind,direction=predict(model,h,o[rows],s[rows]);b=decode(h,o[rows],s[rows]);xy=b['heroes'][:,0,:2].reshape(len(rows),128,2)*8192
            for j,(i,row) in enumerate(chunk):add(groups['dagger_learner_stuck'],h['actors'][i]%2,masks[row],s[i,:,1],direction[j],kind[j],s[i,:,6]<.5,xy[j])
        save(a.output/'status.json',dict(stage='dagger-inference',files_done=number+1,files_total=len(selections),wall_seconds=time.monotonic()-started))
    result=dict(teacher_validation_premid=json.loads((a.output/'teacher-validation.json').read_text()),dagger_learner_stuck={side:finish(value) for side,value in groups['dagger_learner_stuck'].items()},
                dagger_learner_frames=total_learner,stuck_move_frames=stuck,delta_bins_degrees=list(range(-180,180,10)),
                definition='Premid uses exact decision index*0.2 and recorded native arrival elapsed time, not FP16 rounded game time. Direction head conditioned on MOVE; joint_correct also requires argmax action MOVE. Stuck proxy: previous MOVE, no own cast, displacement < 0.5*move_speed*0.2 for >=6 consecutive decisions; FP16 positions. DAgger student-executed frames additionally require recorded teacher_used=0.',
                model=str(a.weights),wall_seconds=time.monotonic()-started)
    save(a.output/'result.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--dagger',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);run(a)
