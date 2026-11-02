"""Re-read a teacher match independently, checking every stored decision."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.demonstrations import demonstration_label,teacher_records
from sf_duel.spec import blank_action

ALLOWED={'item_wraith_band','item_boots','item_bracer','item_tpscroll','item_enchanted_mango',
         'item_flask','item_tango','item_clarity','item_circlet','item_slippers','item_gauntlets',
         'item_recipe_bracer','item_recipe_wraith_band'}


def audit(run):
    run=Path(run);result=json.loads((run/'result.json').read_text())
    previous={};last={};frames=Counter();actions=Counter();purchases=Counter();eligible=Counter()
    outcome=result['outcome'];first_time={};last_time={}
    with (run/result['data'].get('file','frames.jsonl.gz')).open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==result['data']['sha256'],'Data hash mismatch'
    for row in teacher_records(run):
            s=row['observation'];team=s['team'];d=s['demonstration']
            assert s['match']==d['command']['match']
            assert row['previous']==previous.get(team,blank_action()),'Previous action mismatch'
            label,ref=demonstration_label(s,previous.get(team),full_observation_check=True)
            assert row['action']==label and row['target_ref']==ref,'Saved label mismatch'
            if team in last:
                seq,tick=last[team]
                assert s['seq']==seq+1 and d['decision_tick']==tick+d['period_ticks'],'Sequence gap'
            assert abs(30/d['period_ticks']-result['decision_hz'])<.00001
            assert all(r['name'] in ALLOWED for r in s['probe']['inventory']),'Forbidden item'
            assert all(r['level']==0 for r in s['probe']['skill_slots']
                       if r.get('name') in ('nevermore_frenzy','special_bonus_attributes')),'Forbidden skill'
            roster=s['probe']['allied_heroes']
            assert len(roster)==1 and roster[0]['name']=='npc_dota_hero_nevermore','Extra hero'
            last[team]=(s['seq'],d['decision_tick']);previous[team]=label
            frames[str(team)]+=1;actions[str(label['action'])]+=1;purchases[str(label['buy'])]+=1
            first_time.setdefault(str(team),s['time']);last_time[str(team)]=s['time']
            if outcome and s['time']<outcome['time']-.001:eligible[str(team)]+=1
    assert set(frames)=={'2','3'} and sum(frames.values())==result['data']['frames']
    assert dict(actions)==result['data']['actions'] and dict(purchases)==result['data']['purchases']
    assert not result['errors']
    report={'passed':True,'frames':dict(frames),'pre_terminal_frames':dict(eligible),
            'first_game_time':first_time,'last_game_time':last_time,'actions':dict(actions),
            'purchases':dict(purchases),'decision_hz':result['decision_hz'],
            'whole_match_eligible':result['eligible'],'training_started':False}
    (run/'audit.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
    print(json.dumps(audit(args.run)),flush=True)
