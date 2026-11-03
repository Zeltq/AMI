"""Extract concrete episodes from the same SFDUEL02 files used by `replay`."""
import argparse,gzip,json,struct
from pathlib import Path
import numpy as np
from report_p04 import write

def main():
    p=argparse.ArgumentParser();p.add_argument('traces',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    previous={};events={m:[] for m in range(3)};frames={m:0 for m in range(3)};ended=set()
    for file in sorted(a.traces.glob('matches-*.sfduel.gz')):
        raw=gzip.decompress(file.read_bytes());assert raw[:8]==b'SFDUEL02'
        n=struct.unpack_from('<I',raw,8)[0];h=json.loads(raw[12:12+n]);data=np.frombuffer(raw[12+n:],dtype='<f4').reshape(h['frames'],h['matches'],h['width'])
        meta=h['width']-h['unit_capacity']*h['unit_fields']-h['projectile_capacity']*h['projectile_fields']-h['tree_count']-2*h['signal_fields']
        for block in data:
            for m,row in enumerate(block[:3]):
                if m in ended:continue
                units=row[meta:meta+h['unit_capacity']*h['unit_fields']].reshape(h['unit_capacity'],h['unit_fields'])
                signals=row[-2*h['signal_fields']:].reshape(2,h['signal_fields'])
                time=float(row[0])/30;side=m%2;enemy=1-side;u=units[side];e=units[enemy];frames[m]+=1
                old=previous.get(m)
                if old is not None:
                    prior,prior_signals,old_time=old;pu=prior[side];pe=prior[enemy]
                    if 1<=pu[12]<=3 and u[12]==0 and pu[8]-u[8]>1:
                        spell=int(pu[12])-1;distance=[200,450,700][spell]
                        center=u[:2]+np.array([np.cos(u[4]),np.sin(u[4])])*distance
                        hero_distance=float(np.linalg.norm(e[:2]-center))
                        events[m].append({'kind':'raze_completed','time':round(time,3),'side':side,'spell':['near','mid','far'][spell],
                            'hero_center_distance':round(hero_distance,2),'hero_health_before':round(float(pe[2]),2),'hero_health_after':round(float(e[2]),2),
                            'own_lh_delta':int(u[15]-pu[15]),'mana_spent_net':round(float(pu[8]-u[8]),2),'hero_inside_radius_with_hull':hero_distance<=274,
                            'note':'4-tick trace; health/LH deltas can include concurrent projectiles, geometry is sampled after completion'})
                    if u[19]>pu[19]+.2 and pu[13]==enemy:
                        creeps=units[6:228];mask=(creeps[:,7]>0)&(creeps[:,5]!=side)&(creeps[:,6]>=2)
                        distances=np.linalg.norm(creeps[mask,:2]-pu[:2],axis=1)
                        events[m].append({'kind':'hero_attack_cycle','time':round(time,3),'side':side,'hero_distance':round(float(np.linalg.norm(e[:2]-u[:2])),2),'nearest_enemy_creep':round(float(distances.min()),2) if len(distances) else None})
                    tower_id=4 if side==0 else 2;tower=prior[tower_id]
                    away=pu[:2]-tower[:2];move=signals[side,4:6]-64
                    if signals[side,0]==1 and tower[13]==side and np.dot(away,move)>0 and np.linalg.norm(away)<1000:
                        if not events[m] or events[m][-1]['kind']!='tower_retreat' or time-events[m][-1]['time']>3:
                            events[m].append({'kind':'tower_retreat','time':round(time,3),'side':side,'tower_target':int(tower[13]),'distance':round(float(np.linalg.norm(away)),2),'move_bins':[int(v) for v in signals[side,4:6]],'health':round(float(u[2]),2)})
                previous[m]=(units[:6].copy(),signals.copy(),time)
                if row[2]>0 or any(signals[:,9]>0):ended.add(m)
    result={'source':str(a.traces),'matches':[],'limitations':'Trace sampled every four ticks; exact damage, releases and aggro totals come from the event counters in report.json.'}
    for m in range(3):
        kinds=['raze_completed','hero_attack_cycle','tower_retreat']
        result['matches'].append({'match':m,'v10_side':m%2,'frames':frames[m],'ended':m in ended,'counts':{k:sum(e['kind']==k for e in events[m]) for k in kinds},'events':events[m]})
    write(a.output,result)
    for match in result['matches']:print(json.dumps({k:v for k,v in match.items() if k!='events'}))

if __name__=='__main__':main()
