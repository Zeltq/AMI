"""P03: collect 100 unchanged teacher duels for each of two seed groups."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from measure_dota import save


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parallel', type=int, default=20)
    parser.add_argument('--adopt-pid', type=int)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=args.adopt_pid is not None)
    if not args.adopt_pid:(args.run / 'started.json').write_text(json.dumps({'pid': os.getpid()}))
    original = Path('C:/AMI/runs/gpu/ggd0-il-20261102-202637')
    manifest = read_json(original / 'source-manifest.json')
    for name in ('gpu/src/physics.rs', 'gpu/src/scripted_teacher.rs', 'gpu/src/imitation.rs', 'gpu/src/ggd_metrics.rs'):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != manifest[name]:
            raise RuntimeError('Frozen executable source mismatch: ' + name)
    executable = args.run / 'simulator.exe'
    if not args.adopt_pid:shutil.copy2(original / 'ggd-tools.exe', executable)
    expected = manifest['runs\\gpu\\ggd0-il-20261102-202637\\ggd-tools.exe']
    if hashlib.sha256(executable.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen executable checksum mismatch')
    save(args.run / 'source-manifest.json', manifest)
    pilot = Path('C:/AMI/runs/pilot-status.json')
    started = time.monotonic()
    results = []
    for seed in (20261103, 20261104):
        config = args.run / f'collect-{seed}.json'
        output = args.run / f'seed-{seed}'
        save(config, dict(engine='C:/AMI/runs/gpu/dota-calibration-003/engine.json', games=100,
                          parallel_matches=args.parallel, seconds=600, seed=seed, max_blocks=None))
        command = [str(executable), 'collect-teacher', str(config), str(output)]
        seed_started = time.monotonic()
        adopted = args.adopt_pid if seed == 20261103 else None
        with (args.run / f'collect-{seed}.log').open('ab' if adopted else 'wb') as log:
            process = None if adopted else subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                            creationflags=subprocess.CREATE_NO_WINDOW)
            def alive():
                if process is not None:return process.poll() is None
                found=subprocess.run(['tasklist','/FI',f'PID eq {adopted}','/NH'],capture_output=True,text=True,errors='replace')
                return str(adopted) in found.stdout
            while alive():
                path = output / 'status.json'
                progress = read_json(path) if path.exists() else {}
                record = dict(stage='simulator_collect', pid=adopted or process.pid, supervisor_pid=os.getpid(),
                              command=subprocess.list2cmdline(command), run=str(args.run), seed=seed,
                              completed=sum(r['games_recorded'] for r in results), requested=200,
                              progress=progress, wall_minutes=(time.monotonic()-started)/60,
                              background_work_remaining=True, artifacts=[str(output)])
                save(args.run / 'status.json', record)
                current = read_json(pilot); current['P03_simulator'] = record; save(pilot, current)
                time.sleep(5)
        if (process is not None and process.returncode) or not (output / 'report.json').exists():
            record.update(stage='failed', exit_code=process.returncode if process is not None else None, background_work_remaining=False)
            save(args.run / 'status.json', record)
            current = read_json(pilot); current['P03_simulator'] = record; save(pilot, current)
            raise RuntimeError(f'Simulator failed; inspect collect-{seed}.log')
        report = read_json(output / 'report.json')
        results.append(dict(seed=seed, output=str(output), games_recorded=report['games_recorded'],
                            wall_minutes=(time.monotonic()-seed_started)/60, report=report))
        save(args.run / 'results.json', results)
    record.update(stage='simulator_complete', completed=sum(r['games_recorded'] for r in results),
                  wall_minutes=(time.monotonic()-started)/60, background_work_remaining=False)
    save(args.run / 'status.json', record)
    current = read_json(pilot); current['P03_simulator'] = record; save(pilot, current)


if __name__ == '__main__':
    main()
