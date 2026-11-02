"""Wait for the approved dataset, then train once and evaluate once."""
import argparse
from collections import Counter
import hashlib
import json
import msvcrt
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def validate_dataset(dataset, cache, teacher_hash):
    original = json.loads((dataset / 'dataset.json').read_text())
    prepared = json.loads((cache / 'dataset.json').read_text())
    for manifest in (original, prepared):
        matches = manifest['matches']
        if len(matches) != 192 or Counter(m['split'] for m in matches) != {'train': 128, 'validation': 64}:
            raise ValueError('Expected exactly 192 complete matches, split 128/64')
        if len({m['index'] for m in matches}) != 192:
            raise ValueError('Duplicate match indices')
        if manifest['teacher_policy_sha256'] != teacher_hash:
            raise ValueError('Teacher hash changed')
        for m in matches:
            if not m['eligible'] or not m['model_observation_adapter_validated'] or m['reason'] not in ('death', 'tower', 'last_hits'):
                raise ValueError('Incomplete or unvalidated match')
            if m['teacher_policy_sha256'] != teacher_hash:
                raise ValueError('Per-match teacher mismatch')
    if original['matches'] != prepared['matches']:
        raise ValueError('Cache split differs from collected split')
    for m in prepared['matches']:
        metadata = json.loads((cache / f"match-{m['index']:04d}.json").read_text())
        if Path(metadata['run']).resolve() != Path(m['run']).resolve():
            raise ValueError('Cache points to a different match attempt')
        for team in (2, 3):
            file = cache / f"match-{m['index']:04d}-team-{team}.npz"
            if not file.is_file() or not file.stat().st_size:
                raise ValueError(f'Missing prepared episode: {file}')
    return prepared


def linux(path):
    path = Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + '/' + path.as_posix()[3:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(ROOT / 'runs/teacher-pilot-192'))
    parser.add_argument('--cache', default=str(ROOT / 'runs/teacher-pilot-192-cache'))
    parser.add_argument('--output', default=str(ROOT / 'runs/bc-pilot-192'))
    parser.add_argument('--evaluation', default=str(ROOT / 'runs/bc-pilot-192-evaluation'))
    args = parser.parse_args()
    dataset, cache, output, evaluation = map(Path, (args.dataset, args.cache, args.output, args.evaluation))
    state_file = ROOT / 'runs/pilot-followthrough.json'
    def status(stage, **extra):
        value = {'stage': stage, 'updated_unix': time.time(), **vars(args), **extra}
        temp = state_file.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)); temp.replace(state_file)
        pilot_file = ROOT / 'runs/pilot-status.json'
        pilot = json.loads(pilot_file.read_text())
        pilot.update(stage=stage, followthrough_status=str(state_file), training_output=str(output), evaluation_output=str(evaluation))
        if stage in ('training', 'evaluating', 'complete'):
            pilot.update(training_started=True, main_training_started=True)
        if stage == 'training':
            pilot.update(training_pid=extra['process_id'], training_command=extra['command'])
        if stage == 'evaluating':
            pilot.update(evaluation_pid=extra['process_id'], evaluation_command=extra['command'])
        temporary = pilot_file.with_suffix('.followthrough.tmp'); temporary.write_text(json.dumps(pilot, indent=2)); temporary.replace(pilot_file)
        print(json.dumps(value), flush=True)
    # Held for the entire chain: a second coordinator cannot start duplicate jobs.
    with (ROOT / 'runs/pilot-followthrough.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'1'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if output.exists() or evaluation.exists():
                raise FileExistsError('Pilot output exists; refusing to overwrite or silently retrain')
            teacher_hash = json.loads((dataset / 'source-manifest.json').read_text())['teacher_policy_sha256']
            status('waiting_for_dataset')
            while True:
                if (dataset / 'dataset.json').exists() and (cache / 'dataset.json').exists():
                    try:
                        validate_dataset(dataset, cache, teacher_hash)
                        break
                    except json.JSONDecodeError:
                        pass  # Producer may still be finishing its manifest write.
                time.sleep(5)
            if hashlib.sha256((ROOT / 'tools/nativecollect/teacher.go').read_bytes()).hexdigest() != teacher_hash:
                raise ValueError('Working teacher changed since collection')
            output.mkdir()
            train = ['wsl', '-d', 'Ubuntu', '--', 'bash', '/mnt/c/AMI/scripts/wsl_python.sh',
                     '/mnt/c/AMI/scripts/train_bc.py', linux(cache), '--output', linux(output),
                     '--epochs', '5', '--batch-size', '32', '--length', '80', '--burn-in', '16', '--lr', '0.0003', '--minutes', '60']
            with (output / 'train.log').open('wb') as log:
                process = subprocess.Popen(train, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                status('training', process_id=process.pid, command=train)
                if process.wait():
                    raise RuntimeError('Training failed; inspect train.log')
            if not (output / 'complete.json').exists() or not (output / 'best.pt').exists():
                raise RuntimeError('Training exited without a complete checkpoint')
            evaluate = [sys.executable, str(ROOT / 'scripts/evaluate_native_policy.py'), '--checkpoint', str(output / 'best.pt'),
                        '--run', str(evaluation), '--matches', '10', '--workers', '2', '--timescale', '2',
                        '--frozen-bots', str(dataset / 'teacher_bots'), '--teacher-hash', teacher_hash]
            with (output / 'evaluation.log').open('wb') as log:
                process = subprocess.Popen(evaluate, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                status('evaluating', process_id=process.pid, command=evaluate)
                if process.wait():
                    raise RuntimeError('Evaluation failed; inspect evaluation.log')
            result = json.loads((evaluation / 'evaluation.json').read_text())
            if result['matches'] != 10 or result['inference_errors']:
                raise RuntimeError('Evaluation incomplete or had inference errors')
            status('complete', result={k: result[k] for k in ('matches', 'reached_mid', 'attacked_creeps', 'cast_razes')})
        except BaseException as error:
            status('failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
