"""Reproducible P01 dataset counts, death windows and runtime comparisons."""
import argparse,collections,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from ggd_dataset import read_block,decode,layout
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.spec import Act,UNIT_FIELDS

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def ratio(n,d):return {'numerator':int(n),'denominator':int(d)}

def dataset(root,output):
    root=Path(root);manifest=json.loads((root/'dataset.json').read_text());groups,_=layout()
    engine=json.loads((root/'environment/engine.json').read_text());towers=np.asarray(engine['prototypes'])[[2,4],:2]
    acts=np.zeros(len(Act),np.int64);same=0;total=0;zero_previous=0;phases={'pregame':[0,0],'after_zero':[0,0]}
    runs=collections.Counter();pending={};last={};teacher={side:{'frames':0,'minimum_enemy_t1_distance':None,'maximum_progress_from_own_t1':None} for side in ('radiant','dire')}
    fixture_rows=[]
    for index,entry in enumerate(manifest['files']):
        header,obs,sig=read_block(root/entry['file'],entry['sha256']);inv={v:i for i,v in enumerate(header['columns'])}
        at=groups['previous'][0];prev=obs[:,:,[inv[j] for j in range(at,at+len(Act))]]
        times=obs[:,:,inv[groups['global_features'][0]]].astype(np.float32)*600
        positions=obs[:,:,[inv[0],inv[1]]].astype(np.float64)*8192
        valid=sig[:,:,11]>.5;actions=sig[:,:,0].astype(np.int64)
        for i,actor in enumerate(header['actors']):
            pre=valid[i]&(times[i]<0);side=actor%2;name=('radiant','dire')[side];row=teacher[name];row['frames']+=int(pre.sum())
            if pre.any():
                pos=positions[i,pre];ts=times[i,pre];distance=np.linalg.norm(pos-towers[1-side],axis=1)
                direction=towers[1-side]-towers[side];direction/=np.linalg.norm(direction)
                progress=(pos-towers[side])@direction
                for key,values,select,better in [('minimum_enemy_t1_distance',distance,np.argmin,lambda a,b:a<b),('maximum_progress_from_own_t1',progress,np.argmax,lambda a,b:a>b)]:
                    j=int(select(values));old=row[key]
                    if old is None or better(float(values[j]),old['value']):row[key]={'value':float(values[j]),'actor':actor,'game':actor//2,'game_seconds':float(ts[j]),'position':pos[j].tolist()}
            if actor//2%8 in (6,7):
                if actor//2%8==6 and valid[i].all() and (actions[i]==3).sum()>=8 and len(fixture_rows)<2 and actor not in [r['actor'] for r in fixture_rows]:
                    fixture=root/entry['file'];target=Path(output).parent/f'validation-{len(fixture_rows)}.npz'
                    arrays=decode(header,obs[i:i+1],sig[i:i+1]);arrays['resets']=sig[i,:,10].copy();np.savez_compressed(target,**arrays)
                    fixture_rows.append({'actor':actor,'file':str(fixture),'fixture':str(target),'steps':128,'attacks':int((actions[i]==3).sum()),'block_sha256':entry['sha256']})
                continue
            selected=actions[i,valid[i]];acts+=np.bincount(selected,minlength=len(Act));total+=len(selected)
            same+=int(((actions[i]==prev[i].argmax(-1))&valid[i]).sum());zero_previous+=int(((prev[i].sum(-1)==0)&valid[i]).sum())
            for phase,mask in [('pregame',times[i]<0),('after_zero',times[i]>=0)]:
                keep=valid[i]&mask;phases[phase][0]+=int(((actions[i]==0)&keep).sum());phases[phase][1]+=int(keep.sum())
            for j in range(128):
                if not valid[i,j] or sig[i,j,10]>.5:
                    if pending.get(actor,0):runs[pending.pop(actor)]+=1
                if valid[i,j]:
                    if actor in last and float(times[i,j])-last[actor]>.4:
                        if pending.get(actor,0):runs[pending.pop(actor)]+=1
                    if actions[i,j]==0:pending[actor]=pending.get(actor,0)+1
                    elif pending.get(actor,0):runs[pending.pop(actor)]+=1
                    last[actor]=float(times[i,j])
        if (index+1)%128==0:print(json.dumps({'files':index+1,'of':len(manifest['files']),'train_frames':total}),flush=True)
    for length in pending.values():
        if length:runs[length]+=1
    result={'dataset_sha256':hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest(),'files_checked':len(manifest['files']),'train_frames':total,'actions':{a.name:ratio(acts[a.value],total) for a in Act},'action_equals_previous':ratio(same,total),'zero_previous_action_vector':ratio(zero_previous,total),'continue_by_phase':{k:ratio(*v) for k,v in phases.items()},'continue_run_lengths':{'runs':sum(runs.values()),'decisions':sum(k*v for k,v in runs.items()),'minimum':min(runs),'maximum':max(runs),'histogram':dict(sorted(runs.items())),'definition':'consecutive valid decisions per actor; cross 128-step blocks; break at life reset or invalid frame; terminal runs right-censored'},'teacher_pregame_all_splits':teacher,'teacher_t1_positions':towers.tolist(),'validation_fixtures':fixture_rows}
    save(output,result);print(json.dumps({k:v for k,v in result.items() if k!='continue_run_lengths'},indent=2))

def deaths(root,output):
    root=Path(root);header=json.loads((root/'trace-header.json').read_text());columns=header['columns'];actors=collections.defaultdict(list)
    with (root/'trace.jsonl').open() as f:
        for line in f:
            r=json.loads(line);actors[r['actor']].append(r['v'])
    report=json.loads((root/'report.json').read_text());windows=[]
    for match in report['per_match']:
        game=match['game'];radiant=match['sides'][0];death=radiant['first_death_game_seconds']
        if game%2 or death is None or death>=0:continue
        rows=np.asarray(actors[game*2]);time=rows[:,0]/30;keep=(time>=death-15)&(time<=death+4/30);rows=rows[keep];time=time[keep]
        path=Path(output).parent/f'death-game-{game:02}.csv';fields=['game_seconds',*columns[1:25],'top1','p1','top2','p2','top3','p3']
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(fields)
            for t,row in zip(time,rows):
                top=np.argsort(-row[36:47],kind='stable')[:3];writer.writerow([t,*row[1:25],*[v for a in top for v in (Act(int(a)).name,row[36+a])]])
        damage=[]
        for before,after in zip(rows,rows[1:]):
            if after[3]<before[3]:damage.append({'game_seconds':float(after[0]/30),'health_loss_net_regen':float(before[3]-after[3]),'source':int(after[9]),'source_kind':int(after[48]),'source_tier':int(after[49])})
        windows.append({'game':game,'death_game_seconds':death,'rows':len(rows),'csv':str(path),'start_position':rows[0,1:3].tolist(),'end_position':rows[-1,1:3].tolist(),'minimum_enemy_t1_distance':float(rows[:,17].min()),'commands':dict(collections.Counter(Act(int(a)).name for a in rows[:,19])),'orders':dict(collections.Counter(map(int,rows[:,5]))),'damage':damage,'continue_probability_range':[float(rows[:,36].min()),float(rows[:,36].max())]})
        if len(windows)==4:break
    save(output,{'matches':report['matches'],'radiant_learner_pregame_deaths':ratio(len(windows),report['matches']//2),'windows':windows});print(json.dumps(windows,indent=2))

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    for name in ('dataset','deaths'):
        q=s.add_parser(name);q.add_argument('input');q.add_argument('output')
    a=p.parse_args();globals()[a.command](a.input,a.output)

if __name__=='__main__':main()
