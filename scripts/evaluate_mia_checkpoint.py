"""One frozen-checkpoint evaluation after gracefully draining training."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import time

from selfplay_cycle import ROOT, Config, atomic, collect, evaluation_score
from run_mia_restart import alive
from run_mia_autonomous import export_replay, available_dir


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--opponent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--drained-collection', required=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    base, output = Path(args.run).resolve(), Path(args.output).resolve()
    checkpoint, opponent = Path(args.checkpoint).resolve(), Path(args.opponent).resolve()
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint, opponent)}
    cfg = replace(Config.read(base/'config.json'), workers=args.workers, timescale=1., evaluation_matches=20)
    cfg.validate()
    output.mkdir(parents=True, exist_ok=args.resume)
    cfg.write(output/('config-refill.json' if args.resume else 'config.json'))

    def status(phase, **extra):
        atomic(output/'status.json', dict(phase=phase, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint), opponent=str(opponent), workers=cfg.workers, timescale=cfg.timescale, target=20, **extra))

    try:
        status('waiting_for_training_drain', wait_pid=args.wait_pid)
        started = time.monotonic()
        while alive(args.wait_pid):
            if time.monotonic()-started > 900:
                raise RuntimeError('Training did not drain; preserved active games')
            time.sleep(2)
        # Also exclude accidental restarts of the training coordinator.
        with (base/'coordinator.lock').open('a+b') as lock:
            lock.seek(0); lock.write(b'1'); lock.flush(); lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            drained = Path(args.drained_collection)
            previous = read(drained/'collection.json')
            atomic(drained/'exported-replays.json', [export_replay(m['replay']) for m in previous['matches'] if m.get('replay')])
            parts = sorted(output.glob('collection*/collection.json')) if args.resume else []
            if args.resume and not (output/'quality-repair.json').exists():
                raise RuntimeError('Resume requires documented diagnosis')
            games = [m for part in parts for m in read(part)['matches']]
            completed = [m['index'] for m in games]
            if len(completed)!=len(set(completed)) or any(i not in range(1,21) for i in completed):
                raise RuntimeError('Duplicate or invalid evaluation index')
            missing = [i for i in range(1,21) if i not in completed]
            if missing:
                folder = available_dir(output, 'collection-refill') if args.resume else output/'collection'
                status('evaluating', collection=str(folder), completed_prior=len(games), missing_indices=missing)
                games += collect(checkpoint, folder, cfg, [str(opponent)], matches=len(missing), versus_previous=True,
                                 record_prefix=None if args.resume else 'restart_mia2_vs_mia0_20261101_x1', match_indices=missing)
                parts.append(folder/'collection.json')
            raw_parts = [read(part) for part in parts]
            replays = [export_replay(m['replay']) for m in games if m.get('replay')]
            atomic(output/'exported-replays.json', replays)
            if len(games) != 20:
                raise RuntimeError('Evaluation incomplete or quarantined; inspect preserved games before recovery')
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
                raise RuntimeError('Checkpoint changed during evaluation')
            episodes = {}
            for part in parts:
                episodes.update(read(part.parent/'service/stats.json')['episodes'])
            behavior = [episodes[m['run']+':'+str(m['model_team'])] for m in games]
            summary = dict(**evaluation_score(games), checkpoint_sha256=hashes, workers=cfg.workers, timescale=1.,
                           sides={str(side): evaluation_score([m for m in games if m['model_team']==side]) for side in (2, 3)},
                           reached_mid=sum(b['first_mid_time'] is not None for b in behavior),
                           attacked_creeps=sum(b['creep_projectiles']>0 for b in behavior),
                           cast_razes=sum(b['raze_casts']>0 for b in behavior), behaviors=behavior,
                           games=games, replays=replays, wall_seconds=sum(p['wall_seconds'] for p in raw_parts),
                           parts=[str(p.parent) for p in parts], quarantined=sum(len(p.get('quarantine',[])) for p in raw_parts),
                           mixed_concurrency=args.resume)
            atomic(output/'evaluation.json', summary)
            status('complete', score=evaluation_score(games))
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
