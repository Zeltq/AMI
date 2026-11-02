"""Preserve the actual directed mid-lane path_corner chains from the installed map."""
import argparse,hashlib,json
from pathlib import Path
from map_fow import entity_rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',default='runs/gpu/dota-map-entities-001/maps/dota/entities/default_ents.vents');p.add_argument('--output',required=True);a=p.parse_args()
 source=Path(a.entities);rows=list(entity_rows(source));named={r['targetname'].strip('"'):r for r in rows if 'targetname'in r and 'lane_mid_pathcorner_'in r['targetname']};routes={}
 for side,kind in(('radiant','npc_dota_spawner_good_mid'),('dire','npc_dota_spawner_bad_mid')):
  spawner=next(r for r in rows if r.get('classname','').strip('"')==kind);name=spawner['npcfirstwaypoint'].strip('"');chain=[];seen=set()
  while name:
   if name in seen:raise RuntimeError('Loop in authored mid route')
   seen.add(name);r=named[name];x,y,z=json.loads(r['origin']);chain.append({'name':name,'x':x,'y':y,'z':z});name=r['target'].strip('"')
  if len(chain)<3:raise RuntimeError('Incomplete authored route')
  routes[side]={'spawner_origin':json.loads(spawner['origin']),'initial_marker':chain[0],'full_chain':chain,'next_marker_route':[{k:v for k,v in r.items()if k!='name'}for r in chain[1:]]}
 result={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'routes':routes,'scope':'Authored directed paths. Runtime start/arrival tolerance must be checked against native creep trajectories; these values alone do not certify movement.'};Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({s:{'points':len(r['full_chain']),'first_next':r['next_marker_route'][0]}for s,r in routes.items()}))
if __name__=='__main__':main()
