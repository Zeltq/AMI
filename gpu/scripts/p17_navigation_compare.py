"""Compare Dota handoff clocks with an existing P16 navigation trace (no simulation)."""
import argparse
import json
from pathlib import Path
from ggd_dataset import read_block


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('C:/AMI/runs/p16/nav-collect-smoke'));a=p.parse_args()
    manifest=json.loads((a.reference/'dataset.json').read_text(encoding='utf-8'))
    if not manifest.get('scripted_base_to_mid'):raise ValueError('Script-navigation reference required')
    first={};resets={}
    for entry in manifest['files']:
        header,_,signals=read_block(a.reference/entry['file'],entry['sha256']);block=int(entry['file'].split('-')[1])
        for i,actor in enumerate(header['actors']):
            valid=(signals[i,:,11]>.5).nonzero()[0]
            if actor not in first and len(valid):
                first[actor]=block*128+int(valid[0]);resets[actor]=float(signals[i,valid[0],10])
    native=json.loads((a.run/'result.json').read_text(encoding='utf-8'));rows=[]
    for team in (2,3):
        group=[step*.2 for actor,step in first.items() if actor%2==team-2]
        boundary=native['teams'][str(team)]['boundary'][0];handoff=native['teams'][str(team)]['handoffs'][0]
        actual=90+handoff['time']
        rows.append(dict(team=team,sim_arrival_seconds_sum=sum(group),sim_actors=len(group),sim_mean_seconds=sum(group)/len(group),
                         dota_arrival_elapsed_pregame_seconds=actual,dota_actors=1,delta_seconds=actual-sum(group)/len(group),
                         dota_first_observation_relative_seconds=boundary['decision_index']*.2,
                         handoff_count=len(native['teams'][str(team)]['handoffs']),reset=boundary['reset'],
                         previous_action=boundary['input_previous_action'],hp_history_known=boundary['hp_history_known']))
    result=dict(reference=str(a.reference),reference_kind='existing partial navigation diagnostic, not a new CPU model evaluation',
                simulator_first_decisions=first,simulator_handoff_resets=resets,rows=rows,dota=native)
    (a.run/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'rows':rows}))


if __name__=='__main__':main()
