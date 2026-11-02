"""Preserve native initial inputs and known masks from a training recording."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import NativeAdapter,records
from sf_duel.observations import encode

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--output',required=True);a=p.parse_args()
 dataset=json.loads((Path(a.cache)/'dataset.json').read_text());match=next(m for m in sorted(dataset['matches'],key=lambda m:m['index'])if m['split']=='train')
 source=Path(match['run'])/'native-decisions.pb.gz';first={};models={}
 for row in records(source):
  if row.HasField('model_supplement'):models[(row.model_supplement.team,row.model_supplement.tick)]=row.model_supplement
  if row.HasField('decision')and row.decision.team not in first:first[row.decision.team]=row
  if len(first)==2 and all((t,r.decision.observed_tick)in models for t,r in first.items()):break
 if len(first)!=2:raise RuntimeError('Missing initial observations')
 sides={}
 for team,row in first.items():
  snapshot=NativeAdapter().snapshot(row,models[(team,row.decision.observed_tick)]);arrays=encode(snapshot).arrays
  sides[str(team)]={'time':snapshot['time'],'hero':snapshot['self'],'items':snapshot['items'],'skills':snapshot['skills'],'towers':snapshot['towers'],
   'hero_features':arrays['heroes'][0].tolist(),'tower_features':arrays['towers'].tolist(),'item_features':arrays['items'].tolist(),
   'action_mask':arrays['action_mask'].tolist(),'attack_mask':arrays['attack_mask'].tolist(),
   'free_tp_ready_dota_time':snapshot['time']+snapshot['items'][3].get('cooldown',0)}
 report={'source':str(source),'split':'train','sides':sides,'scope':'Initial state and encoding contract, not environment dynamics or evaluation data'}
 Path(a.output).write_text(json.dumps(report,indent=2));print(a.output)

if __name__=='__main__':main()
