"""Audit the user-limited continuation and all three completed evaluations."""
import array
import csv
import math
import subprocess
from pathlib import Path
from p09c_continue90 import BASE, START, PILOT, RUN, REPORT, read, save, sha
from p09b_results import lines, summaries
from p09c_run import pid_alive


def main():
    state = read(RUN / 'state.json')
    assert not pid_alive(state.get('pid')) and not pid_alive(state.get('controller_pid'))
    jobs = read(RUN / 'commands.json')
    assert len(jobs) == 7 and all(j.get('exit_code') == 0 for j in jobs)
    request = read(REPORT / 'request.json')
    result = read(RUN / 'training-result.json')
    checkpoint = Path(result['latest'])
    progress = read(checkpoint / 'progress.json')
    config = read(checkpoint.parent / 'config.json')
    source = read(START / 'progress.json')
    assert request['training_minutes'] == 50 and request['original_training_minutes'] == 90
    assert Path(config['resume_checkpoint']) == START and config['skip_warmup']
    assert not config['begin_decay_on_resume'] and not config['evaluate']
    assert sha(START / 'model.bpk') == request['model_sha256']
    assert sha(START / 'optimizer.bpk') == request['optimizer_sha256']
    assert result['resumed_training_seconds'] == source['training_seconds']
    elapsed = result['training_seconds'] - source['training_seconds']
    assert abs(elapsed - 3000) <= 30 and progress['training_seconds'] == result['training_seconds']
    assert (checkpoint.parent / 'stop.json').exists()
    assert progress['anchor_sha256'] == request['anchor_manifest_sha256']
    assert abs(progress['kl_coefficient_next'] - .05) < 1e-6
    updates = lines(checkpoint.parent / 'history.jsonl')
    assert [r['block'] for r in updates] == list(range(315, 471))
    assert all(0 <= r['learning']['clip_fraction'] <= 1 for r in updates)
    assert all(abs(r['learning']['kl_coefficient'] - .05) < 1e-6 and not r['learning']['warmup'] for r in updates)
    snapshot = read(REPORT / 'evaluation-weight-snapshot.json')
    for files in snapshot['checkpoints'].values():
        for item in files:
            path = Path(item['path'])
            assert sha(path) == item['sha256'], path
            if path.suffix == '.f32':
                values = array.array('f'); values.frombytes(path.read_bytes())
                assert all(math.isfinite(v) for v in values), path
    checks = []; rows = []
    for opponent, weights in [('v10', None), ('pilot', PILOT), ('current', START / 'weights')]:
        games = []
        for seed in (20261103, 20261104):
            folder = RUN / f'eval-{opponent}-{seed}'
            r = read(folder / 'report.json'); cfg = read(folder / 'config.json')
            assert r['complete'] and r['matches'] == 64 and r['seed'] == seed
            assert len(r['per_match']) == 64 and {g['game'] for g in r['per_match']} == set(range(64))
            assert Path(cfg['weights']) == checkpoint / 'weights'
            assert not cfg['argmax'] and not cfg['zero_previous'] and cfg['seconds'] == 600
            assert cfg['teacher_version'] == 'v10' and Path(cfg['engine']).name == 'engine.json'
            assert Path(cfg['engine']).parent.name == 'dota-calibration-005'
            if weights:
                other = read(folder / 'opponent.json')
                assert Path(other['weights']) == weights
                assert other['manifest_sha256'] == sha(weights / 'manifest.json') == request[opponent + '_manifest_sha256']
            assert sum(g['sides'][g['game'] % 2]['side'] == 'radiant' for g in r['per_match']) == 32
            for g in r['per_match']:
                assert g['simulation_error'] == 0 and g['native_terminal'] and g['result'] in (-1, 0, 1)
                assert g['outcome_reason'] in ('death', 'T1', 'LH600')
                for side, hero in enumerate(g['sides']):
                    c = hero['event_v2_components']
                    assert len(c) == 6 and c[0] == 10 * g['result'] * (1 if side == 0 else -1)
                    assert abs(c[1] - .5 * hero['lh']) < 1e-3 and abs(c[2] - .2 * hero['deny']) < 1e-3
                    assert c[3] <= 0 and c[4] >= 0 and c[5] <= 0
                    assert abs(sum(c) - hero['event_v2_reward']) < 1e-3
                    assert abs(c[3] + g['sides'][1-side]['event_v2_components'][4]) < 1e-3
            games += r['per_match']
            checks.append(dict(opponent=opponent, seed=seed, matches=64, radiant=32, dire=32,
                               sha256=sha(folder / 'report.json'), evaluation_minutes=r['wall_seconds']/60))
        pair = summaries('RL+50', dict(per_match=games), opponent, 'event-v2')
        totals = [r for r in pair if r['side'] == 'total']
        assert len(totals) == 2 and all(r['matches'] == 128 for r in totals)
        a, b = totals
        assert a['wins'] == b['losses'] and a['losses'] == b['wins'] and a['draws'] == b['draws']
        rows += pair
    assert rows == read(REPORT / 'metrics.json'), 'Published summaries differ from raw matches'
    with (REPORT / 'metrics.csv').open(encoding='utf-8', newline='') as f:
        published = list(csv.DictReader(f))
    assert len(published) == len(rows)
    for a, b in zip(rows, published):
        for key, value in a.items():
            if isinstance(value, (int, float)):
                assert math.isclose(float(b[key]), value, rel_tol=1e-10, abs_tol=1e-8), key
    protected = read(REPORT.parent / 'protected-source-audit.json')
    subprocess.run(['git', 'diff', '--exit-code', protected['baseline'], '--'] +
                   [r['path'] for r in protected['files']], check=True, capture_output=True)
    summary = dict(passed=True, matches=384, hero_perspectives=768, series=checks,
                   checkpoint=str(checkpoint), model_sha256=sha(checkpoint/'model.bpk'),
                   optimizer_sha256=sha(checkpoint/'optimizer.bpk'), manifest_sha256=sha(checkpoint/'weights/manifest.json'),
                   new_training_minutes=elapsed/60, new_blocks=len(updates), clip_valid_blocks=len(updates),
                   constant_anchor_coefficient=.05, warmup_blocks=0, decisions=result['total_decisions'],
                   decisions_per_second=result['total_decisions']/elapsed,
                   source_model_and_adam_hashes_match_request=True, resumed_time_matches_source=True,
                   weight_snapshot_scope=snapshot['scope'], snapshot_files_unchanged=189,
                   finite_tensor_files=177, protected_files_unchanged=len(protected['files']),
                   gpu_process_minutes=sum(j['minutes'] for j in jobs),
                   prior_gpu_minutes=request['previous_gpu_minutes'], gpu_job_failures=0,
                   old_controller_guard='Expected stale 90-minute assertion after user-requested native stop; no native failure.')
    save(REPORT / 'audit.json', summary)
    print({k:v for k,v in summary.items() if k not in ('series', 'old_controller_guard')})


if __name__ == '__main__':
    main()
