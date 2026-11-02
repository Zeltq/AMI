"""User-authorized 30-minute BC continuation followed by accelerated evaluation."""
import json
import msvcrt
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'runs/bc-continue-30m'
    evaluation = ROOT / 'runs/bc-continue-30m-evaluation'
    def status(stage, **extra):
        value = {'stage': stage, 'updated_unix': time.time(), 'training': str(out), 'evaluation': str(evaluation), **extra}
        p = ROOT / 'runs/bc-continuation-status.json'; tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2)); tmp.replace(p)
        print(json.dumps(value), flush=True)
    with (ROOT / 'runs/bc-continuation.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'1'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if out.exists() or evaluation.exists():
            raise FileExistsError('Continuation output already exists')
        out.mkdir()
        try:
            train = ['wsl', '-d', 'Ubuntu', '--', 'bash', '/mnt/c/AMI/scripts/wsl_python.sh', '/mnt/c/AMI/scripts/train_bc.py',
                     '/mnt/c/AMI/runs/teacher-pilot-192-cache', '--output', '/mnt/c/AMI/runs/bc-continue-30m',
                     '--resume', '/mnt/c/AMI/runs/bc-pilot-192/best.pt', '--epochs', '10000', '--minutes', '30']
            with (out / 'train.log').open('wb') as log:
                process = subprocess.Popen(train, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                status('training', pid=process.pid, command=train)
                if process.wait(): raise RuntimeError('Continuation training failed')
            if not (out / 'complete.json').exists(): raise RuntimeError('Training completion missing')
            readiness = ROOT / 'runs/continuation-executor-ready.json'
            if not readiness.exists():
                status('waiting_for_executor_verification')
                while not readiness.exists(): time.sleep(5)
            command = [sys.executable, str(ROOT / 'scripts/evaluate_native_policy.py'), '--checkpoint', str(out / 'best.pt'),
                       '--run', str(evaluation), '--matches', '10', '--workers', '4', '--worker', '10',
                       '--timescale', '2', '--offset', '6', '--allow-late', '--record-first', 'sf_model_bc30m_20261030',
                       '--frozen-bots', str(ROOT / 'runs/teacher-pilot-192/teacher_bots')]
            with (out / 'evaluation.log').open('wb') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                status('evaluating', pid=process.pid, command=command)
                if process.wait(): raise RuntimeError('Continuation evaluation failed')
            status('complete')
        except BaseException as error:
            status('failed', error=str(error))
            raise


if __name__ == '__main__': main()
