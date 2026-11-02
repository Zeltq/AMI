"""Reproducible P01 dataset counts, death windows and runtime comparisons."""
import argparse,collections,csv,hashlib,json,math,subprocess,sys
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
                if actor//2%8==6 and valid[i].all() and (actions[i]==3).sum()>=8 and len(fixture_rows)<2 and actor%2 not in [r['actor']%2 for r in fixture_rows]:
                    fixture=root/entry['file'];target=Path(output).parent/f'validation-{len(fixture_rows)}.npz'
                    arrays=decode(header,obs[i:i+1],sig[i:i+1]);arrays['resets']=sig[i,:,10].copy();np.savez_compressed(target,**arrays)
                    fixture_rows.append({'actor':actor,'file':str(fixture),'fixture':str(target),'steps':128,'attacks':int((actions[i]==3).sum()),'block_sha256':entry['sha256']})
                continue
            selected=actions[i,valid[i]];acts+=np.bincount(selected,minlength=len(Act));total+=len(selected)
            same+=int(((actions[i]==prev[i].argmax(-1))&valid[i]).sum());zero_previous+=int(((prev[i].sum(-1)==0)&valid[i]).sum())
            for phase,mask in [('pregame',times[i]<0),('after_zero',times[i]>=0)]:
                keep=valid[i]&mask;phases[phase][0]+=int(((actions[i]==0)&keep).sum());phases[phase][1]+=int(keep.sum())
            for j in range(128):
                if not valid[i,j]:
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
    result={'dataset_sha256':hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest(),'files_checked':len(manifest['files']),'train_frames':total,'actions':{a.name:ratio(acts[a.value],total) for a in Act},'action_equals_previous':ratio(same,total),'zero_previous_action_vector':ratio(zero_previous,total),'continue_by_phase':{k:ratio(*v) for k,v in phases.items()},'continue_run_lengths':{'runs':sum(runs.values()),'decisions':sum(k*v for k,v in runs.items()),'minimum':min(runs),'maximum':max(runs),'histogram':dict(sorted(runs.items())),'definition':'consecutive valid decisions per actor; cross 128-step blocks including BPTT resets; break at invalid frame or time gap; terminal runs right-censored'},'teacher_pregame_all_splits':teacher,'teacher_t1_positions':towers.tolist(),'validation_fixtures':fixture_rows}
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
        live=rows[time<death];live_time=time[time<death]
        path=Path(output).parent/f'death-game-{game:02}.csv';fields=['game_seconds',*columns[1:25],'top1','p1','top2','p2','top3','p3']
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(fields)
            for t,row in zip(live_time,live):
                top=np.argsort(-row[36:47],kind='stable')[:3];writer.writerow([t,*row[1:25],*[v for a in top for v in (Act(int(a)).name,row[36+a])]])
        damage=[]
        for before,after in zip(rows,rows[1:]):
            if after[3]<before[3]:damage.append({'game_seconds':float(after[0]/30),'health_loss_net_regen':float(before[3]-after[3]),'source':int(after[9]),'source_kind':int(after[48]),'source_tier':int(after[49])})
        windows.append({'game':game,'death_game_seconds':death,'rows':len(live),'csv':str(path),'start_position':live[0,1:3].tolist(),'end_position':live[-1,1:3].tolist(),'minimum_enemy_t1_distance':float(live[:,17].min()),'commands':dict(collections.Counter(Act(int(a)).name for a in live[:,19])),'move_coordinates':dict(collections.Counter(f'{int(r[23])},{int(r[24])}' for r in live if r[19]==1)),'orders':dict(collections.Counter(map(int,live[:,5]))),'damage':damage,'move_probability_range':[float(live[:,37].min()),float(live[:,37].max())],'stop_probability_max':float(live[:,40].max())})
        if len(windows)==4:break
    save(output,{'matches':report['matches'],'radiant_learner_pregame_deaths':ratio(len(windows),report['matches']//2),'windows':windows});print(json.dumps(windows,indent=2))

def parity_fixture(checkpoint,fixture,output,device='cuda'):
    import torch
    from sf_duel.model import Policy
    from sf_duel.observations import packed_tensors
    from train_ggd import recurrent
    from weights import write_bundle
    torch.set_num_threads(2);torch.manual_seed(20261102)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=Policy(combat_inputs=True).eval().to(device);model.load_state_dict(saved['state_dict'])
    data=np.load(fixture);arrays={k:data[k] for k in data.files if k not in ('labels','loss_valid','resets')};b=packed_tensors(arrays,device);y=torch.as_tensor(data['labels'],device=device);resets=data['resets'].reshape(1,-1)
    with torch.no_grad():
        h,keys=recurrent(model,b,1,resets);action=y[:,0];is_item=action==Act.ITEM;item=torch.where(is_item,y[:,2],torch.zeros_like(y[:,2]));embedding=model.action_embed(action);condition=torch.cat((h,embedding,model.item_embed(item)),-1)
        logits={'action':model.action_head(h),'buy':model.buy_head(h),'item':model.item_head(torch.cat((h,embedding),-1)),'target':torch.einsum('bu,bnu->bn',model.query(condition),keys)/8,'x':model.x_head(condition),'y':model.y_head(condition)}
        masks={'action':b['action_mask'],'buy':b['buy_mask'],'item':torch.where(is_item[:,None],b['item_mask'],torch.ones_like(b['item_mask'])),'x':torch.ones_like(logits['x'],dtype=torch.bool),'y':torch.ones_like(logits['y'],dtype=torch.bool)}
        target_mask=torch.where((action==Act.ATTACK)[:,None],b['attack_mask'],b['self_mask']);target_mask=torch.where((is_item&(item==3))[:,None],b['tp_mask'],target_mask);target_mask=torch.where((is_item&(item==6))[:,None],b['tree_mask'],target_mask);masks['target']=target_mask
        result={k:v.cpu().numpy() for k,v in b.items()};result.update(labels=y.cpu().numpy(),resets=resets.reshape(-1),expected_h=h.cpu().numpy(),expected_value=model.value_head(h).squeeze(-1).cpu().numpy());result.update({f'expected_{k}_logits':v.cpu().numpy() for k,v in logits.items()});result.update({f'{k}_mask':v.cpu().numpy() for k,v in masks.items()})
    runtime={'torch':torch.__version__,'hip':torch.version.hip,'device':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}
    write_bundle(Path(output),result,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),source=str(fixture),forward='train_ggd.recurrent, batch_sequences=1, length=128, life resets applied',runtime=runtime)
    print(json.dumps({'output':str(output),'steps':len(y),'runtime':runtime}))

def observations(collect,evaluation,output):
    collect=Path(collect);evaluation=Path(evaluation);cm=json.loads((collect/'dataset.json').read_text());em=json.loads((evaluation/'recording.json').read_text())
    counts=collections.defaultdict(lambda:[0,0,0.]);signals_equal=0;signals_count=0;previous_equal=0;previous_count=0;resets_equal=0;resets_count=0;first=None;blocks=0;last_actions={};boundary_correct=0;boundary_count=0
    if len(cm['files'])!=len(em['files']):raise ValueError('Recording block counts differ')
    for ce,ee in zip(cm['files'],em['files']):
        ch,co,cs=read_block(collect/ce['file'],ce['sha256']);eh,eo,es=read_block(evaluation/ee['file'],ee['sha256'])
        if ch!=eh:raise ValueError('Recording headers differ: '+ce['file'])
        valid=(cs[:,:,11]>.5)&(es[:,:,11]>.5);ca=decode(ch,co,cs);ea=decode(eh,eo,es);keep=valid.reshape(-1)
        for key in ca:
            x=ca[key][keep];y=ea[key][keep];equal=x==y;n=int(equal.size);m=int((~equal).sum());counts[key][0]+=m;counts[key][1]+=n
            if n:counts[key][2]=max(counts[key][2],float(np.abs(x.astype(np.float64)-y.astype(np.float64)).max()))
            if m and first is None:first={'file':ce['file'],'tensor':key,'mismatches':m}
        equal=co.view(np.uint16)==eo.view(np.uint16);counts['packed_fp16_all_frames'][0]+=int((~equal).sum());counts['packed_fp16_all_frames'][1]+=equal.size
        signals_equal+=int((cs==es).sum());signals_count+=cs.size;resets_equal+=int((cs[:,:,10]==es[:,:,10]).sum());resets_count+=cs[:,:,10].size
        previous=ca['previous'].reshape(len(ch['actors']),128,-1)[:,:,:len(Act)].argmax(-1)
        for i in range(len(ch['actors'])):
            actor=ch['actors'][i];expected=np.r_[last_actions.get(actor,0),cs[i,:-1,0]].astype(np.int64);select=valid[i];previous_equal+=int(((previous[i]==expected)&select).sum());previous_count+=int(select.sum());last_actions[actor]=int(cs[i,-1,0])
            boundaries=np.zeros(128,dtype=bool);boundaries[0]=True;boundary_correct+=int((((cs[i,:,10]>.5)==boundaries)&select).sum());boundary_count+=int(select.sum())
        blocks+=1
    result={'blocks':blocks,'tensors':{k:{'mismatches':ratio(v[0],v[1]),'max_absolute_delta':v[2]} for k,v in counts.items()},'signals_equal':ratio(signals_equal,signals_count),'resets_equal':ratio(resets_equal,resets_count),'previous_is_prior_issued_action':ratio(previous_equal,previous_count),'periodic_reset_matches_128_step_boundaries':ratio(boundary_correct,boundary_count),'first_difference':first,'seed':em['seed'],'comparison':'same one-game teacher-v9 rollout, FP16 bits including padding; decoded valid inputs also compared per tensor; initial previous action is canonical CONTINUE'}
    save(output,result);print(json.dumps(result,indent=2))

def evaluation_grid(config):
    config=json.loads(Path(config).read_text());run=(ROOT/'runs/p01').resolve();rows=[]
    for model in config['models']:
        for selection in ('sampling','argmax'):
            job=f"eval-{model['name']}-{selection}";output=run/job/'evaluation'
            command=[config['executable'],'evaluate-ggd',config['engine'],model['weights'],str(output),'--mode','teacher','--matches',str(config['matches']),'--seconds',str(config['seconds']),'--seed',str(config['seed']),'--'+selection]
            if model.get('zero_previous'):command.append('--zero-previous')
            subprocess.run([sys.executable,str(Path(__file__).with_name('p01_job.py')),job,'--',*command],cwd=ROOT,check=True)
            report=json.loads((output/'report.json').read_text())
            if not report['complete']:raise RuntimeError('Incomplete evaluation: '+job)
            for side in report['summary']['sides']:
                row={k:side[k] for k in ('controller','side','games','lh','deny','deaths','first_death_game_seconds')};row.update(model=model['name'],selection=selection,source=str(output/'report.json'))
                row['games_with_death']=sum(1 for g in report['per_match'] for s in g['sides'] if s['controller']==side['controller'] and s['side']==side['side'] and s['deaths']>0)
                row['pregame_deaths']=sum(1 for g in report['per_match'] for s in g['sides'] if s['controller']==side['controller'] and s['side']==side['side'] and s['first_death_game_seconds'] is not None and s['first_death_game_seconds']<0)
                rows.append(row)
            save(run/(config['name']+'-summary.json'),rows);print(json.dumps({'model':model['name'],'selection':selection,'summary':report['summary']['sides']}),flush=True)

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    for name in ('dataset','deaths'):
        q=s.add_parser(name);q.add_argument('input');q.add_argument('output')
    q=s.add_parser('parity-fixture');q.add_argument('checkpoint');q.add_argument('fixture');q.add_argument('output');q.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    q=s.add_parser('observations');q.add_argument('collect');q.add_argument('evaluation');q.add_argument('output')
    q=s.add_parser('evaluation-grid');q.add_argument('config')
    a=p.parse_args()
    if a.command=='parity-fixture':parity_fixture(a.checkpoint,a.fixture,a.output,a.device)
    elif a.command=='observations':observations(a.collect,a.evaluation,a.output)
    elif a.command=='evaluation-grid':evaluation_grid(a.config)
    else:globals()[a.command](a.input,a.output)

if __name__=='__main__':main()
