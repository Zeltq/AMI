"""Run the P03 Dota cohort once on two existing worker slots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from measure_dota import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', nargs='+', type=int, default=[18, 19])
    parser.add_argument('--seeds', nargs=2, type=int, default=[20261103, 20261104])
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=False)
    (args.run / 'started.json').write_text(json.dumps({'pid': os.getpid()}))
    started = time.monotonic()
    results = []
    active = {}
    pilot = Path('C:/AMI/runs/pilot-status.json')

    def worker(slot):
        rows = []
        for index in range(args.workers.index(slot), 20, len(args.workers)):
            seed = args.seeds[index // 10]
            run = args.run / f'match-{index:02d}'
            command = [sys.executable, '-u', str(Path(__file__).with_name('measure_dota.py')),
                       '--run', str(run), '--worker', str(slot), '--seconds', '600', '--timescale', '4',
                       '--wall-timeout', '600', '--seed', str(seed), '--no-pilot-update']
            with (args.run / f'match-{index:02d}.log').open('wb') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                active[slot] = dict(index=index, seed=seed, run=str(run), pid=process.pid, command=subprocess.list2cmdline(command))
                code = process.wait()
            row = dict(index=index, seed=seed, worker=slot, run=str(run), exit_code=code)
            rows.append(row)
            results.append(row)
            active.pop(slot, None)
            if code:
                raise RuntimeError(f'P03 match {index} failed; inspect {run}')
        return rows

    with ThreadPoolExecutor(max_workers=len(args.workers)) as pool:
        futures = [pool.submit(worker, slot) for slot in args.workers]
        while not all(f.done() for f in futures):
            jobs = []
            for item in list(active.values()):
                item = item.copy()
                path = Path(item['run']) / 'status.json'
                if path.exists():
                    child = json.loads(path.read_text(encoding='utf-8'))
                    item.update(game_seconds=child.get('game_seconds'), dota_pid=child.get('dota_pid'), stage=child['stage'])
                jobs.append(item)
            elapsed = time.monotonic() - started
            record = dict(stage='dota_series', pid=os.getpid(), command=subprocess.list2cmdline([sys.executable, *sys.argv]),
                          run=str(args.run), completed=len(results), requested=20, wall_minutes=elapsed / 60,
                          matches_per_hour=len(results) / max(elapsed, 1) * 3600, errors=sum(r['exit_code'] != 0 for r in results),
                          retries=0, jobs=jobs, background_work_remaining=True)
            save(args.run / 'status.json', record)
            current = json.loads(pilot.read_text(encoding='utf-8')); current['P03'] = record; save(pilot, current)
            save(args.run / 'matches.json', sorted(results, key=lambda r: r['index']))
            time.sleep(2)
        for future in futures:
            future.result()
    save(args.run / 'matches.json', sorted(results, key=lambda r: r['index']))
    record.update(stage='dota_complete', completed=len(results), jobs=[], wall_minutes=(time.monotonic()-started)/60, background_work_remaining=False)
    save(args.run / 'status.json', record)
    current = json.loads(pilot.read_text(encoding='utf-8')); current['P03'] = record; save(pilot, current)


if __name__ == '__main__':
    main()
