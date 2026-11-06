"""Funded short dedicated matches to inspect purchases and courier delivery."""
import argparse
import json
import msvcrt
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from play_vs_chio import below_normal,save
from p17_matches import match
from p17_p16_guard import check


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seconds',type=float,default=45)
    p.add_argument('--require-boots',action='store_true');p.add_argument('--delivery',action='store_true');a=p.parse_args();below_normal();check(force=True)
    runtime=Path('C:/AMI/runs/p17/bridge-001/runtime')
    setup="for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do h:ModifyGold(2000,true,DOTA_ModifyGold_Unspecified) end\n"
    if a.delivery:
        setup+='''local owner=Entities:FindByClassname(nil,'npc_dota_hero_nevermore')
owner:SetContextThink('p17_delivery_test',function()
 if GameRules:GetDOTATime(false,true)<-55 then return .1 end
 for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do
  for slot=0,16 do local it=h:GetItemInSlot(slot);if it and it:GetName()=='item_boots' then h:RemoveItem(it) end end
  h:ModifyGold(500,true,DOTA_ModifyGold_Unspecified)
 end
 print('SF_CHIO_ECONOMY_REDELIVERY|'..tostring(GameRules:GetDOTATime(false,true)))
 return nil
end,.1)
'''
    with Path('C:/AMI/runs/p17/runtime.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        result=match(a.run,Path('C:/AMI/runs/p17/models/ppo-final.pt'),2,a.seconds,runtime,setup_lua=setup)
    rows=[];forced=None
    for line in (a.run/'dota.log').open(errors='replace'):
        if 'SF_CHIO_ECONOMY|' in line:rows.append(json.loads(line.split('SF_CHIO_ECONOMY|',1)[1]))
        if 'SF_CHIO_ECONOMY_REDELIVERY|' in line:forced=float(line.split('SF_CHIO_ECONOMY_REDELIVERY|',1)[1])
    teams={}
    for team in (2,3):
        values=[r for r in rows if r['team']==team]
        boots=[r for r in values if any(i['name']=='item_boots' and i['slot']<=5 for i in r['inventory'])]
        attempts=[r for r in values if r['kind']=='attempt']
        teams[str(team)]=dict(boots_equipped=bool(boots),first_equipped_time=boots[0]['time'] if boots else None,
            boots_attempts=[r for r in attempts if r['item']=='item_boots'],last=values[-1] if values else None,
            equipped_after_redelivery=bool(forced is not None and any(r['time']>forced+1 for r in boots)))
    out=dict(funded_gold_per_hero=2000,redelivery_time=forced,diagnostic_not_model_evaluation=True,teams=teams,events=len(rows),match_result=result)
    save(a.run/'economy-result.json',out);save(a.run/'economy-events.json',rows)
    print(json.dumps(out,ensure_ascii=False))
    if a.require_boots:assert all(t['boots_equipped'] and any(r['success'] for r in t['boots_attempts']) for t in teams.values())
    if a.delivery:assert forced is not None and all(t['equipped_after_redelivery'] and any(r['time']>=forced and r['success'] for r in t['boots_attempts']) for t in teams.values())


if __name__=='__main__':main()
