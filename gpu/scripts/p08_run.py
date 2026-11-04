"""P08 subprocess controller: durable commands, PID, budget and shared status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('C:/AMI/runs/p08')
REPO = Path(__file__).resolve().parents[2]

def save(path, value):
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def shared(value):
    path = ROOT.parent / 'pilot-status.json'
    current = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    current['P08'] = value
    save(path, current)

def run(stage, command):
    state_path = ROOT / 'controller.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    history = previous.get('history', [])
    started = time.time()
    artifact_dir = Path(command[4]) if len(command)>4 and command[1]=='evaluate-ggd' else ROOT/stage
    with (ROOT / f'{stage}.stdout.log').open('w') as out, (ROOT / f'{stage}.stderr.log').open('w') as err:
        child = subprocess.Popen(command, cwd=REPO, stdout=out, stderr=err)
        state = dict(stage=stage, controller_pid=os.getpid(), child_pid=child.pid,
                     command=subprocess.list2cmdline(command), argv=command,
                     artifacts=str(artifact_dir), started_unix=started, history=history)
        save(state_path, state)
        while child.poll() is None:
            state['elapsed_seconds'] = time.time() - started
            status = artifact_dir / 'status.json'
            if status.exists():
                try: state['progress'] = json.loads(status.read_text())
                except (OSError, json.JSONDecodeError): pass
            save(state_path, state)
            shared(state)
            time.sleep(10)
        state['returncode'] = child.returncode
    history.append(dict(stage=stage, command=state['command'], argv=command,
                        wall_seconds=time.time()-started, returncode=child.returncode))
    state.update(stage='complete' if child.returncode == 0 else 'failed', history=history)
    save(state_path, state)
    shared(state)
    if child.returncode:
        raise SystemExit(child.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['zero-lr', 'warmup', 'smoke', 'main', 'tests', 'build', 'full-tests', 'v3-smoke', 'prepare', 'smoke-eval', 'main-prepare'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    # The controller is locked for its full lifetime; no duplicate jobs.
    import msvcrt
    with (ROOT / 'controller.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        stages = (['build', 'zero-lr', 'warmup', 'v3-smoke', 'smoke'] if args.stage == 'prepare' else ['full-tests', 'build', 'main'] if args.stage == 'main-prepare' else [args.stage])
        for stage in stages:
            execute(stage)

def execute(stage):
    if stage == 'main' and (ROOT/'pause.json').exists():
        raise SystemExit('Main PPO is paused by the user; remove pause.json only after explicit permission to resume.')
    if stage == 'smoke-eval':
        result = json.loads((ROOT/'smoke/result.json').read_text(encoding='utf-8'))
        cfg = json.loads((ROOT/'smoke/config.json').read_text(encoding='utf-8'))
        for selection in ('sampling', 'argmax'):
            command = [str(ROOT/'target/release/sf-duel-gpu.exe'), 'evaluate-ggd', cfg['engine'], str(Path(result['latest'])/'weights'), str(ROOT/f'smoke/eval-final-{selection}'), '--mode', 'teacher', '--matches', '64', '--seconds', '600', '--seed', '20261103', f'--{selection}', '--trace']
            run(f'smoke-eval-{selection}', command)
        subprocess.run([os.sys.executable, str(REPO/'gpu/scripts/p08_results.py')], cwd=REPO, check=True)
        return
    if stage in ('tests', 'full-tests', 'build'):
        os.environ['SF_DUEL_DATA_ROOT'] = 'C:/AMI'
        command = ['cargo', 'build' if stage == 'build' else 'test', '--release', '--locked', '--manifest-path', str(REPO/'gpu/Cargo.toml'), '--target-dir', str(ROOT/'target')]
        if stage == 'tests': command += ['p08_', '--', '--nocapture', '--test-threads=1']
        if stage == 'full-tests': command += ['--', '--test-threads=1']
    else:
        # Include all successful AND interrupted training observed in status files.
        used = 0.0
        for path in ROOT.glob('*/status.json'):
            data = json.loads(path.read_text())
            used += data.get('result', {}).get('training_seconds', data.get('latest', {}).get('training_seconds', data.get('training_seconds', 0)))
        config = json.loads((REPO/f'reports/P08/{stage}.json').read_text())
        if used + config['minutes']*60 > 350*60:
            raise SystemExit(f'PPO budget guard: {used/60:.2f} minutes used; requested {config["minutes"]}')
        command = [str(ROOT/'target/release/sf-duel-gpu.exe'), 'train-ggd', str(REPO/f'reports/P08/{stage}.json'), str(ROOT/stage)]
    run(stage, command)
    if stage in ('zero-lr', 'warmup'):
        subprocess.run([str(Path(os.sys.executable)), str(REPO/'gpu/scripts/p08_results.py'), '--verify', stage], cwd=REPO, check=True)
    if stage == 'smoke':
        subprocess.run([str(Path(os.sys.executable)), str(REPO/'gpu/scripts/p08_results.py')], cwd=REPO, check=True)

if __name__ == '__main__':
    main()
