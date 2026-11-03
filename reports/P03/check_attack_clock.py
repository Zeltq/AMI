"""Observe own attack clocks against first visible native attack projectiles."""
import argparse
import gzip
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sf_duel.protos.worldstate_pb2 import CMsgBotWorldState


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={}
    for team in (2,3):
        seen=set();shots=[];changes=[];last=None
        with gzip.open(a.run/f'world-{team}.pb.gz','rb')as f:
            while header:=f.read(4):
                state=CMsgBotWorldState.FromString(f.read(struct.unpack('<I',header)[0]))
                own=next((u for u in state.units if u.team_id==team and u.name=='npc_dota_hero_nevermore'),None)
                if own is None:continue
                if last is not None and own.last_attack_time>last:changes.append(state.dota_time)
                last=own.last_attack_time
                for unit in state.units:
                    for shot in unit.incoming_tracking_projectiles:
                        if shot.caster_handle==own.handle and shot.is_attack and shot.handle not in seen:
                            seen.add(shot.handle);shots.append(dict(dota_time=state.dota_time,game_time=state.game_time,last_attack_time=own.last_attack_time,delay=state.game_time-own.last_attack_time))
        result[str(team)]=dict(attack_clock_changes=len(changes),observed_projectiles=len(shots),minimum_delay=min((s['delay']for s in shots),default=None),maximum_delay=max((s['delay']for s in shots),default=None),samples=shots)
    report=dict(sides=result,scope='Read-only 90s diagnostic: first observed own projectiles and protobuf last_attack_time. Sampling and visibility can delay or omit projectiles; this does not validate the whole Lua forecast.')
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n');print(json.dumps({t:{k:v for k,v in s.items()if k!='samples'}for t,s in result.items()}))


if __name__=='__main__':main()
