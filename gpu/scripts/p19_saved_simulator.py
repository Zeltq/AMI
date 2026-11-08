"""Read existing P16/P18 records on CPU; never constructs or advances a world."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import struct
import numpy as np
from ggd_dataset import read_block,layout

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path('C:/AMI/runs/p19/simulator')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    reports={
        'p16-dagger2-v9':Path('C:/AMI/runs/p16/eval-sampling-dagger-2-v9-sampling/report.json'),
        'p16-selfplay-v9':Path('C:/AMI/runs/p16/eval-self-play-120m-v9-sampling/report.json'),
        'p18-selfplay360-v9':Path('C:/AMI/runs/p18/eval-self-play-360m-v9-sampling/report.json'),
    }
    summary={};minutes=[]
    for name,path in reports.items():
        data=json.loads(path.read_text());games=data['per_match'];sides=[s for g in games for s in g['sides']]
        summary[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'matches':len(games),'sides':len(sides),'controllers':{}}
        for controller in sorted({s['controller'] for s in sides}):
            rows=[s for s in sides if s['controller']==controller]
            summary[name]['controllers'][controller]={'denominator':len(rows),
                **{key:sum(s.get(key,0) for s in rows) for key in ['gold','gold_received','gold_spent','lh','deny']},
                'purchases':{item:sum(s.get('purchases',{}).get(item,0) for s in rows) for item in ['wraith_band','boots','bracer']}}
        for game in games:
            for side in game['sides']:
                for snap in side.get('minute_snapshots',[]):
                    minutes.append(dict(source=name,game=game['game'],side=side['side'],controller=side['controller'],**snap))
    (OUTPUT/'minute-snapshots.json').write_text(json.dumps(minutes,indent=2))
    # A fixed, reproducible sample: both sides of games 0..15, all saved blocks.
    dataset=Path('C:/AMI/runs/p16/sampling-dagger-2-256')
    manifest=json.loads((dataset/'dataset.json').read_text());engine=json.loads((dataset/'environment/engine.json').read_text())
    selected=[e for e in manifest['files'] if e['file'].endswith('shard-000.ggd.gz')]
    histories={i:[] for i in range(32)};sources=[]
    for entry in selected:
        h,o,signals=read_block(dataset/entry['file'],entry['sha256']);groups,_=layout(h['version'])
        inverse={c:i for i,c in enumerate(h['columns'])}
        def group(name):
            offset,shape,_=groups[name]
            indices=[inverse[i] for i in range(offset,offset+int(np.prod(shape)))]
            return o[:,:,indices].astype(np.float32).reshape(*o.shape[:2],*shape)
        heroes=group('heroes');items=group('items');times=group('global_features')[:,:,0]*600
        for index,actor in enumerate(h['actors']):
            if actor not in histories:continue
            end=manifest['game_metrics'][actor//2]['game_seconds']
            for frame in range(h['horizon']):
                t=float(times[index,frame]);own=heroes[index,frame,0];inv=items[index,frame]
                if t>end+.3:continue
                level=int(round(float(own[23])*15));needed=float(own[25])*2000
                cumulative=sum(engine['experience'][:level])-needed
                histories[actor].append(dict(time=t,gold=float(own[24])*2000,xp=cumulative,
                    level=level,lh=round(float(own[26])*150),deny=round(float(own[27])*50),
                    owned=[round(float(inv[i,:4].sum())*10) for i in range(8)]))
        sources.append({'file':entry['file'],'sha256':entry['sha256']})
    samples=[];purchase_intervals=[]
    for actor,rows in histories.items():
        for target in range(-90,601,10):
            row=min(rows,key=lambda r:abs(r['time']-target))
            if abs(row['time']-target)<=.35:samples.append(dict(game=actor//2,side=actor%2,target_time=target,**row))
        for item,name in enumerate(['wraith_band','boots','bracer']):
            first=next((i for i,r in enumerate(rows) if r['owned'][item]>0),None)
            purchase_intervals.append(dict(game=actor//2,side=actor%2,item=name,
                state='initial_grant' if first==0 else 'purchased' if first is not None else 'not_observed',
                after_time=rows[first-1]['time'] if first else None,
                by_time=rows[first]['time'] if first is not None else None))
    (OUTPUT/'samples-10s-fp16.json').write_text(json.dumps(samples,indent=2))
    (OUTPUT/'purchase-intervals-fp16.json').write_text(json.dumps(purchase_intervals,indent=2))
    summary['saved_observation_sample']={'dataset':str(dataset),'source':'dagger-round-2, v10 teacher/model mixture; not v9 evaluation',
        'games':16,'sides':32,'files':sources,'samples':len(samples),'frames':sum(map(len,histories.values())),
        'precision':'FP16 normalized observations: approximate gold, XP and time; purchase intervals include quantization.',
        'initial_gold_values':sorted({r[0]['gold'] for r in histories.values()}),
        'purchase_counts':{name:sum(r['item']==name and r['state']=='purchased' for r in purchase_intervals) for name in ['wraith_band','boots','bracer']}}
    # Exact float32 reward snapshots complement the normalized FP16 policy input.
    reward_histories={i:[] for i in range(32)};reward_sources=[];purchase_events=[];passive=[];denied_xp=[]
    for path in sorted(dataset.glob('chio-reward-*.f32.gz')):
        with gzip.open(path,'rb') as f:
            size=struct.unpack('<I',f.read(4))[0];header=json.loads(f.read(size))
            n,horizon,width=header['shape'];fields=header['snapshot_fields'];snap=len(fields)
            values=np.frombuffer(f.read(32*horizon*width*4),dtype='<f4').reshape(32,horizon,width)
        reward_sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for actor in range(32):
            for frame in range(header['frames']):
                before=values[actor,frame,:snap];after=values[actor,frame,snap:2*snap]
                if before[21]>0:continue
                keep=['game_seconds','gold_received','gold_current','xp_total','last_hits','denies','gold_spent']
                reward_histories[actor].append({k:float(before[fields.index(k)]) for k in keep})
                if after[21]>0:reward_histories[actor].append({k:float(after[fields.index(k)]) for k in keep})
                enemy_before=values[actor^1,frame,:snap];enemy_after=values[actor^1,frame,snap:2*snap]
                if enemy_after[13]-enemy_before[13]==1 and after[3]>before[3]:
                    denied_xp.append(float(after[3]-before[3]))
                for item,name in enumerate(['wraith_band','boots','bracer']):
                    col=fields.index('purchases_'+str(item))
                    if after[col]>before[col]:purchase_events.append(dict(actor=actor,item=name,after_time=float(before[0]),by_time=float(after[0]),count=float(after[col]-before[col])))
                if before[0]>=0 and after[12]==before[12] and after[14]==before[14] and after[15]==before[15] and after[21]==0:
                    dt=float(after[0]-before[0]);gain=float(after[1]-before[1])
                    if dt>0:passive.append(dict(actor=actor,start=float(before[0]),seconds=dt,gold= gain,gold_per_minute=gain/dt*60))
    reward_samples=[]
    for actor,values in reward_histories.items():
        for target in range(-90,601,10):
            nearest=min(values,key=lambda r:abs(r['game_seconds']-target))
            if abs(nearest['game_seconds']-target)<.11:reward_samples.append(dict(actor=actor,target_time=target,**nearest))
    for name,data in [('samples-10s-f32',reward_samples),('purchase-intervals-f32',purchase_events),('passive-intervals-f32',passive)]:
        (OUTPUT/(name+'.json')).write_text(json.dumps(data,indent=2))
    summary['reward_record_sample']={'sides':32,'samples':len(reward_samples),'sources':reward_sources,
        'initial_gold_values':sorted({r[0]['gold_current'] for r in reward_histories.values()}),
        'purchase_counts':dict((name,sum(e['count'] for e in purchase_events if e['item']==name)) for name in ['wraith_band','boots','bracer']),
        'passive_gold':sum(r['gold'] for r in passive),'passive_seconds':sum(r['seconds'] for r in passive),
        'xp_on_opponent_deny_values':sorted(set(denied_xp)),
        'passive_gpm_min':min(r['gold_per_minute'] for r in passive),'passive_gpm_max':max(r['gold_per_minute'] for r in passive)}
    summary['missing_fields']=['Per-transaction gold reason and reliable/unreliable split',
        'Exact creep identity/type linked to each LH/deny bounty',
        'Exact purchase transaction tick inside 0.2s reward transitions (bounded F32 intervals supplied)',
        '10-second snapshots and complete pregame in P18 evaluation reports',
        '10-second snapshots for selected P16-v9 evaluation matches (different collection sample supplied)']
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='saved_observation_sample'},indent=2))
    print(json.dumps({k:v for k,v in summary['saved_observation_sample'].items() if k!='files'},indent=2))


if __name__=='__main__':main()
