"""Index completed, audited teacher trajectories without starting training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from audit_run import audit,read

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('runs',nargs='+');args=parser.parse_args();entries=[]
    for directory in args.runs:
        run=Path(directory).resolve();result=json.loads((run/'run-result.json').read_text())
        assert result['mode']=='teacher' and result['outcome'] and result['outcome']['reason'] in ('death','tower','last_hits')
        assert not result['stats']['errors'] and not result['stats']['api_errors']
        report=audit(run)
        assert report['passed'],report['failures']
        accepted={};observations=0
        for s in read(run/'snapshots.jsonl.gz'):
            observations+=1;e=s.get('execution')
            if isinstance(e,dict) and 'seq' in e:
                accepted[(s['team'],e['seq'])]=bool(e.get('accepted') and not e.get('purchase_rejected')
                    and all(v==-1 for v in e.get('purchase',[])))
        keys=[];match=None
        for d in read(run/'decisions.jsonl.gz'):
            assert d['controller']=='teacher';match=d['match']
            terminal_time=result['outcome']['time']
            if accepted.get((d['team'],d['seq']),False) and d['time']<terminal_time-.001:keys.append([d['team'],d['seq']])
        eligible=run/'eligible-decisions.jsonl.gz'
        with gzip.open(eligible,'wt',encoding='utf-8') as f:
            for team,seq in keys:f.write(json.dumps({'match':match,'team':team,'seq':seq})+'\n')
        source=json.loads((run/'source-manifest.json').read_text())
        files={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in
            ('snapshots.jsonl.gz','decisions.jsonl.gz','eligible-decisions.jsonl.gz','source-manifest.json')}
        if (run/'source.zip').exists():files['source.zip']=hashlib.sha256((run/'source.zip').read_bytes()).hexdigest()
        entries.append({'run':str(run),'match':match,'teacher_version':source['teacher_version'],
            'outcome':result['outcome'],'observations':observations,'eligible_decisions':len(keys),
            'sides':report['sides'],'sha256':files})
    destination=Path(args.output).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    manifest={'spec_version':1,'purpose':'warm-start candidate trajectories; teacher strength not certified',
        'training_started':False,'split_unit':'whole match, including both sides',
        'eligible_label_rule':'pre-terminal decision, accepted executor ACK and successful purchase; unacknowledged actions excluded',
        'matches':entries,'total_matches':len(entries),'total_eligible_decisions':sum(e['eligible_decisions'] for e in entries)}
    destination.write_text(json.dumps(manifest,indent=2));print(json.dumps({'manifest':str(destination),
        'matches':len(entries),'eligible_decisions':manifest['total_eligible_decisions']}),flush=True)

if __name__=='__main__':main()
