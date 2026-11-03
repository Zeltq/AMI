import gzip
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sf_duel.protos.worldstate_pb2 import CMsgBotWorldState


class RecordedAlignmentTests(unittest.TestCase):
    def test_same_tick_matching_counts_a_real_health_difference(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);lines=[]
            for team in (2,3):
                world=CMsgBotWorldState(team_id=team,dota_time=1)
                u=world.units.add(name='npc_dota_hero_nevermore',team_id=team,health=100,mana=50,action_type=3)
                u.location.x=10;u.location.y=20
                payload=world.SerializeToString()
                with gzip.open(root/f'world-{team}.pb.gz','wb')as f:f.write(struct.pack('<I',len(payload))+payload)
                lines.append('SF_P03_OBSERVATION|'+json.dumps({'snapshot':{'team':team,'time':1,'p03_action_type':3,'self':{'health':99 if team==2 else 100,'mana':50,'x':10,'y':20}}}))
            (root/'lua.txt').write_text('\n'.join(lines))
            subprocess.run([sys.executable,str(ROOT/'diagnostics/check_native_alignment.py'),'--lua-log',str(root/'lua.txt'),'--world-root',str(root),'--output',str(root/'report.json')],check=True,capture_output=True)
            report=json.loads((root/'report.json').read_text())
            self.assertEqual(report['matched_frames'],{'2':1,'3':1})
            self.assertEqual(report['checks']['health'],{'equal':1,'different':1,'missing':0,'total':2})
            self.assertEqual(report['checks']['x']['equal'],2)
