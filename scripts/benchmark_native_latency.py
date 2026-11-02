"""Short, separated timing conditions using the same checkpoint and teacher."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'runs/policy-latency-grid'
    base.mkdir(exist_ok=False)
    results = []
    for speed, offset in ((2, 6), (2, 7), (4, 6), (4, 7)):
        run = base / f'x{speed}-t{offset}'
        command = [sys.executable, str(ROOT / 'scripts/evaluate_native_policy.py'),
                   '--checkpoint', str(ROOT / 'runs/bc-pilot-192/best.pt'), '--run', str(run),
                   '--matches', '2', '--workers', '2', '--worker', '14', '--until', '90',
                   '--timescale', str(speed), '--offset', str(offset), '--max-attempts', '1',
                   '--frozen-bots', str(ROOT / 'runs/teacher-pilot-192/teacher_bots')]
        started = time.monotonic()
        with (base / f'x{speed}-t{offset}.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            (base / 'active.json').write_text(json.dumps({'timescale': speed, 'offset': offset, 'pid': process.pid, 'command': command}))
            code = process.wait()
        if code:
            (base / 'failure.json').write_text(json.dumps({'timescale': speed, 'offset': offset, 'returncode': code}))
            raise RuntimeError(f'Timing condition failed: {run}')
        evaluation = json.loads((run / 'evaluation.json').read_text())
        models = [m['sides'][str(m['model_team'])] for m in evaluation['outcomes']]
        decisions = sum(m['decisions'] for m in models)
        misses = sum(m['missed_deadlines'] for m in models)
        row = {'timescale': speed, 'offset_ticks': offset, 'workers': 2, 'decisions': decisions,
               'missed_deadlines': misses, 'miss_percent': 100 * misses / max(1, decisions),
               'wall_seconds': time.monotonic() - started,
               'per_match': [{'index': m['index'], 'team': m['model_team'], 'game_seconds': m['time'],
                              'decisions': m['sides'][str(m['model_team'])]['decisions'],
                              'misses': m['sides'][str(m['model_team'])]['missed_deadlines']} for m in evaluation['outcomes']]}
        results.append(row)
        temporary = base / 'results.tmp'; temporary.write_text(json.dumps(results, indent=2)); temporary.replace(base / 'results.json')
        print(json.dumps(row), flush=True)
    (base / 'complete.json').write_text(json.dumps({'conditions': len(results)}))


if __name__ == '__main__':
    main()
