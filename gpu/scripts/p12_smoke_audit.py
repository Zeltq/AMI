"""Publish the two-match P12 smoke gate with hashes of local raw evidence."""
import hashlib
import json
from pathlib import Path
from p10_results import dota

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('C:/AMI/runs/p12/native-smoke')
OUT=ROOT/'reports/P12'


def main():
    result=dota(RUN)
    assert len(result['games'])==2
    assert sorted(g['team'] for g in result['games'])==[2,3]
    for game in result['games']:
        q=game['transport'];n=q['checked_before_terminal']
        assert game['mid_game_time'] is not None
        assert game['actions'].get(3,0)>0 and game['hero_attack_commands']>0 and game['raze_casts']>0
        assert n==q['acknowledged'] and not q['unacknowledged'] and not q['late_executions']
        assert q['rejected']/n<.02
        source=Path(game['source'])
        game['source_sha256']={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ('dota.log','native-decisions.pb.gz','duel-result.json','match.json')}
    status=json.loads((RUN/'status.json').read_text())
    stats=json.loads((RUN/'service/stats.json').read_text())
    config=json.loads((RUN/'config.json').read_text())
    assert status['stage']=='complete' and status['completed']==2 and not stats['errors']
    assert stats['requests']==sum(g['transport']['decisions'] for g in result['games'])
    assert hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest()==config['checkpoint_sha256']
    result.update(gate_passed=True,config=config,status=status,service=stats,
        transport_scope='Only model commands; Go commands for the Lua teacher side are not executed.',
        terminal_scope='First death ends duel; later projectile kills excluded. Pending terminal commands excluded from ACK denominator.')
    path=OUT/'dota-smoke.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    assert path.stat().st_size<1_000_000
    print(json.dumps(dict(matches=2,gate_passed=True,wall_minutes=status['wall_minutes'],
        acknowledged=sum(g['transport']['acknowledged'] for g in result['games']),
        rejected=sum(g['rejected'] for g in result['games']))))


if __name__=='__main__':main()
