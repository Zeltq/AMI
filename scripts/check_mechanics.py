"""Separate cheat-enabled API test; its observations are never demonstrations."""
import json
import argparse
from pathlib import Path
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game
from sf_duel.live.bridge import LogPackets,publish

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(ROOT/'runs/mechanics-check-006'))
    parser.add_argument('--from-log');parser.add_argument('--levels',action='store_true')
    args=parser.parse_args();run=Path(args.from_log or args.run).resolve();run.mkdir(exist_ok=True)
    if not args.from_log and (run/'dota.pid').exists():raise RuntimeError('Mechanics test directory already used')
    latest={};report={};seen={2:set(),3:set()};acks={}
    def observe(s):
        team=s['team'];latest[team]=s
        e=s.get('execution')
        if isinstance(e,dict) and e.get('purchase'):acks[(team,e['seq'])]=e
        seen[team].update(row['name'] for row in s['probe']['inventory'])
    def finished():
        return set(latest)=={2,3} and all(s['next_permanent']==2 and len(seen[t])>=8
            and (not args.levels or (s['self']['level']>=10 and s['probe']['skill_index']>=11)) for t,s in latest.items())
    if args.from_log:
        for s in LogPackets(run/'dota.log').read():observe(s)
        assert finished(),'Recorded mechanics state does not meet requested assertions'
    else:
      with Game(run,uuid.uuid4().hex,4,15,diagnostic_bot_level=12 if args.levels else None) as game:
        packets=LogPackets(game.log);deadline=time.monotonic()+75;prepared=False
        while time.monotonic()<deadline:
            for s in packets.read():
                observe(s);team=s['team']
                buy=1 if s['next_permanent']<2 and s['legal']['buys'][1] else 0
                for choice,name in ((2,'item_tpscroll'),(3,'item_enchanted_mango'),(4,'item_flask'),(5,'item_tango'),(6,'item_clarity')):
                    if buy==0 and name not in seen[team] and s['legal']['buys'][choice]:buy=choice
                publish(s['player'],dict(match=s['match'],seq=s['seq'],action=0,buy=buy,observed_time=s['time']),ROOT/'bots')
            if packets.errors:
                (run/'level-result.json').write_text(json.dumps({'passed':False,'errors':packets.errors},indent=2))
                raise RuntimeError(f'Late skill implementation rejected: {packets.errors[-1]}')
            if set(latest)=={2,3} and not prepared:
                game.command('dota_bot_give_gold 5000');prepared=True
            if finished():break
            time.sleep(.002)
        else:raise TimeoutError('Mechanics did not reach skill/purchase assertions')
    for team,s in latest.items():
            skills={r['name']:r['level'] for r in s['probe']['skill_slots'] if r.get('name')}
            assert skills['nevermore_frenzy']==0
            if args.levels:
                assert all(skills[name]==4 for name in ('nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_dark_lord'))
                assert skills['nevermore_requiem']==1 and skills['special_bonus_unique_nevermore_7']==1
                assert skills.get('special_bonus_unique_nevermore_4',0)==0
                assert skills.get('special_bonus_attributes',0)==0
                assert s['probe']['ability_points']==s['self']['level']-10
            assert s['next_permanent']==2
            report[team]={'level':s['self']['level'],'skills':skills,'purchased_items':sorted(seen[team]),
                'next_permanent':s['next_permanent'],'api_errors':s.get('errors')}
            assert not s.get('errors')
    output={'passed':True,'scope':'purchases_and_skills' if args.levels else 'purchases_and_frenzy_exclusion',
        'late_skill_build_verified':args.levels,'diagnostic_cheats':True,'eligible_demonstration':False,'sides':report,
        'purchase_acknowledgements':len(acks),'talent_ability':'special_bonus_unique_nevermore_7',
        'talent_screen_side_verified':True,'talent_verification_source':'https://www.dota2.com/hero/shadowfiend',
        'talent_verification_detail':'Official hero data lists unique_nevermore_7 first; official React UI renders even talent indices on Right.'}
    (run/('level-result.json' if args.levels else 'purchase-result.json')).write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)

if __name__=='__main__':main()
