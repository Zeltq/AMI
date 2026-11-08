import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu/scripts'))
from p19_analyze_dota import analyze,creep_type


def test_flagbearer_is_a_distinct_named_creep():
    assert creep_type({'victim':'npc_dota_creep_goodguys_flagbearer'})=='flagbearer'


def test_balances_reconcile_purchases_and_leave_ambiguous_lh_unassigned(tmp_path):
    def state(team,t,gold,reliable,lh=0):
        return dict(kind='state',team=team,time=t,gold=gold,reliable=reliable,
                    unreliable=gold-reliable,xp=0,lh=lh,deny=0,deaths=0,inventory=[])
    rows=[state(2,0,100,10),state(3,0,100,10),
          state(2,1,36,11),  # 65 spent, one periodic gold.
          state(2,2,71,12,1),  # 34 creep gold plus one periodic gold.
          state(2,3,7,13,2),  # simultaneous purchase and LH: attribution unsafe.
          dict(kind='kill',time=2,victim='npc_dota_creep_badguys_melee',team=3,
               attacker='npc_dota_hero_nevermore',attacker_team=2),
          state(3,3,104,14)]
    (tmp_path/'events.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    buys=[dict(kind='attempt',team=2,time=1,item='item_enchanted_mango',gold_before=101,gold=36,success=True),
          dict(kind='attempt',team=2,time=3,item='item_flask',gold_before=107,gold=7,success=True)]
    (tmp_path/'dota.log').write_text('\n'.join('SF_CHIO_ECONOMY|'+json.dumps(x) for x in buys))
    result=analyze(tmp_path)
    assert result['reward_by_type']['melee_lh']['gold']==34
    assert result['attributed']==1 and result['credits']==2
    assert result['teams']['2']['passive_observed_gold']==2
    assert result['teams']['2']['samples']==0
    assert len(result['teams']['2']['missing_times'])==70


def test_same_timestamp_purchase_after_lh_uses_log_order(tmp_path):
    def state(team,t,gold,lh=0):
        return dict(kind='state',team=team,time=t,gold=gold,reliable=10,
                    unreliable=gold-10,xp=0,lh=lh,deny=0,deaths=0,inventory=[])
    rows=[state(2,0,100),state(3,0,100),
          dict(kind='kill',time=1,victim='npc_dota_creep_badguys_melee',team=3,
               attacker='npc_dota_hero_nevermore',attacker_team=2),
          state(2,1,138,1),state(2,1,38,1)]
    purchase=dict(kind='attempt',team=2,time=1,item='item_flask',gold_before=138,gold=38,success=True)
    lines=['SF_P19|'+json.dumps(row) for row in rows]
    lines.insert(4,'SF_CHIO_ECONOMY|'+json.dumps(purchase))
    (tmp_path/'dota.log').write_text('\n'.join(lines))
    (tmp_path/'events.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    result=analyze(tmp_path)
    assert result['attributed']==result['credits']==1
    assert result['reward_by_type']['melee_lh']['gold']==38
    assert result['teams']['2']['passive_observed_gold']==0
