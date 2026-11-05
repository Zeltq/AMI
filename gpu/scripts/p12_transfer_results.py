"""Publish complete P12 Dota/005 transfer cohorts and death causes."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from p10_results import dota,simulator
from play_vs_model import events

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('C:/AMI/runs/p12')
OUT=ROOT/'reports/P12'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    data=json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n'
    assert len(data.encode('utf-8'))<1_000_000,name
    (OUT/name).write_text(data,encoding='utf-8',newline='\n')


def main():
    native=dota(RUN/'native-eight');sim=simulator(RUN/'sim-v9/report.json')
    status=json.loads((RUN/'native-eight/status.json').read_text())
    sim_job=json.loads((RUN/'sim-v9-job.json').read_text())
    assert status['stage']=='complete' and sim_job['stage']=='complete' and sim['complete']
    assert len(native['games'])==8 and len(sim['games'])==64
    for g in native['games']:
        death=g['telemetry']['first_death'];cause=None
        if g['deaths']:
            assert death and death['team']==g['team']
            killer=death.get('killer','')
            hits=[r for r in events(Path(g['source'])/'dota.log') if r['kind']=='hurt'
                  and r['target']=='npc_dota_hero_nevermore' and r['target_team']==g['team']
                  and abs(r['time']-death['time'])<1/30+.001]
            ability=hits[-1]['ability'] if hits else None
            cause='tower' if 'tower' in killer else 'creeps' if 'creep' in killer else 'raze' if ability and 'shadowraze' in ability else 'hero_attack' if killer=='npc_dota_hero_nevermore' and hits and not ability else killer
        g['death_causes']={cause:g['deaths']} if cause else {}
        q=g['transport']
        assert q['acknowledged']==q['checked_before_terminal'] and not q['late_executions'] and not q['unacknowledged']
        g['source_sha256']={name:sha(Path(g['source'])/name) for name in ('dota.log','native-decisions.pb.gz','match.json')}
    for g in sim['games']:
        side=next(s for s in g['both_sides'] if s['controller']=='GGD')
        g['death_causes']={c['cause']:int(c['count']) for c in side['death_causes'] if c['count']}
        assert sum(g['death_causes'].values())==g['deaths']
    for series,n in ((native,4),(sim,32)):
        for team in (2,3):
            games=[g for g in series['games'] if g['team']==team];s=series['summary'][str(team)]
            assert len(games)==n and s['wins']+s['losses']+s['draws']==n
            s['death_causes']=dict(sum((Counter(g['death_causes']) for g in games),Counter()))
            s['win_reasons']=dict(Counter(g['reason'] for g in games if g['outcome']=='win'))
            s['loss_reasons']=dict(Counter(g['reason'] for g in games if g['outcome']=='loss'))
    config=json.loads((RUN/'native-eight/config.json').read_text())
    assert sha(Path(config['checkpoint']))==config['checkpoint_sha256']
    history_path=RUN/'native-eight/resume-history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    services=[json.loads(p.read_text()) for p in sorted((RUN/'native-eight').glob('service*/stats.json'))]
    assert all(not s['errors'] for s in services)
    assert sum(s['requests'] for s in services)==sum(g['transport']['decisions'] for g in native['games'])
    provenance=dict(export_sha256=sha(Path(config['checkpoint'])),
        simulator_sha256=sha(Path('C:/AMI/runs/p11/sf-duel-gpu.exe')),
        engine_sha256=sha(Path('C:/AMI/runs/p07/dota-calibration-005/engine.json')),
        manifest_sha256=sha(Path('C:/AMI/runs/p11/train-001/checkpoint-000644/weights/manifest.json')),
        native_config=config,native_status=status,sim_job=sim_job,sim_config=json.loads((RUN/'sim-v9/config.json').read_text()),
        resume_history=history,services=services)
    write('dota-eight.json',native);write('simulator.json',sim)
    write('transfer-summary.json',dict(dota=native['summary'],simulator=sim['summary'],provenance=provenance))
    commands=[dict(stage='native',command=h['prior_status']['command'],minutes=h['prior_status']['wall_minutes'],error=h['prior_status']['error']) for h in history]
    commands.append(dict(stage='native-final',command=status['command'],cumulative_minutes=status['wall_minutes']))
    commands.append(dict(stage='simulator',command=sim_job['command'],minutes=sim_job['minutes'],exit_code=sim_job['exit_code']))
    write('commands.json',commands)
    print(json.dumps(dict(dota=native['summary'],simulator=sim['summary']),indent=2))


if __name__=='__main__':main()
