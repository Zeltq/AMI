"""Compact final P08 metrics, command provenance and training-window diagnostics."""
import csv
import json
from pathlib import Path
import p08_revised as runner
import p08_results as results

ROOT, OUT = runner.ROOT, runner.REPORT


def main():
    stage = runner.read(OUT / 'stage-rev-main.json')
    assert stage['returncode'] == 0 and not stage['timed_out']
    result = stage['result']
    history = results.lines(ROOT / 'rev-main/history.jsonl')
    episodes = results.lines(ROOT / 'rev-main/episodes.jsonl')
    assert len(history) == result['blocks']
    evaluations = []
    side_rows = []
    for path in sorted((ROOT / 'rev-main').glob('eval-*/report.json')):
        v = results.evaluation(path)
        assert v['complete'] and v['matches'] == 64
        evaluations.append(v)
        for side in v['sides']:
            row = dict(evaluation=path.parent.name, **side)
            actions = row.pop('actions')
            row['action_decisions'] = sum(actions)
            row.update({f'action_{i}': n for i, n in enumerate(actions)})
            row['farm'] = row['lh'] + row['deny']
            side_rows.append(row)
    assert len(evaluations) == 4
    with (OUT / 'final-evaluation-sides.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(side_rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(side_rows)
    windows = []
    for start in range(0, 120, 20):
        blocks = [h for h in history if start * 60 < h['training_seconds'] <= (start + 20) * 60 + (60 if start == 100 else 0)]
        completed = [e for e in episodes if e['block'] in {h['block'] for h in blocks}]
        windows.append(dict(start_minute=start, end_minute=start + 20, blocks=len(blocks),
                            decisions=sum(h['learning']['valid_decisions'] for h in blocks),
                            mean_ev_post0=sum(h['learning']['ev_post0'] for h in blocks) / len(blocks),
                            mean_ev_pregame=sum(h['learning']['ev_pregame'] for h in blocks) / len(blocks),
                            completed_episodes=len(completed),
                            episodes_with_initialization_prefix=sum(e.get('initialization_prefix_decisions', 0) > 0 for e in completed),
                            reward_sum=sum(e['reward'] for e in completed),
                            lh=sum(e['components'][0] for e in completed),
                            deny=sum(e['components'][1] * 2 for e in completed)))
    training_stages = []
    for name in ['rev-critic-2', 'rev-async-trial-1e5', 'rev-async-trial-3e5', 'rev-main']:
        record = runner.read(OUT / f'stage-{name}.json')
        training_stages.append(dict(stage=name, result=record['result'],
                                    learner_decisions_per_second=record['learner_decisions_per_second'],
                                    sampled_peak_memory=record['sampled_peak_memory']))
    budget = runner.budget()
    assert budget['used_seconds'] <= budget['limit_seconds']
    assert sum(w['decisions'] for w in windows) == result['total_decisions']
    assert sum(w['completed_episodes'] for w in windows) == result['episodes']
    diagnostics = {k: dict(min=min(h['learning'][k] for h in history),
                           max=max(h['learning'][k] for h in history),
                           first5_mean=sum(h['learning'][k] for h in history[:5]) / 5,
                           last5_mean=sum(h['learning'][k] for h in history[-5:]) / 5)
                   for k in ['ev_post0', 'ev_pregame', 'value_loss', 'rollout_kl_m2', 'clip_fraction', 'entropy']}
    final = dict(result=result, evaluations=evaluations, training_stages=training_stages,
                 total_core_training_decisions=sum(t['result']['total_decisions'] for t in training_stages),
                 total_core_completed_episodes=sum(t['result']['episodes'] for t in training_stages),
                 diagnostics=diagnostics, windows=windows, budget=budget,
                 mixed_phase_blocks=sum(h['phase_histogram_verified'] and all(v > 0 for v in h['phase_decisions']) for h in history),
                 early_stop_blocks=sum(h['learning']['early_stop'] for h in history),
                 note='Window EV is a mean of per-block diagnostics, not pooled explained variance. Episode windows use completion time and may include frozen initialization prefixes.')
    runner.save(OUT / 'final-metrics.json', final)
    commands = [dict(stage=r['stage'], command=r['command'], argv=r['argv'], returncode=r['returncode'])
                for p in sorted(OUT.glob('stage-*.json')) for r in [runner.read(p)]]
    runner.save(OUT / 'result-commands.json', dict(processes=commands,
                embedded_main_evaluations=dict(parent_command=stage['command'],
                    config='reports/P08/revised/main.json', matches=64, modes=['sampling', 'argmax'], seed=20261103,
                    reports=[str(p) for p in sorted((ROOT / 'rev-main').glob('eval-*/report.json'))])))
    print(json.dumps(dict(budget=budget, diagnostics=diagnostics, windows=windows,
                          total_core_training_decisions=final['total_core_training_decisions'],
                          total_core_completed_episodes=final['total_core_completed_episodes']), ensure_ascii=False))


if __name__ == '__main__':
    main()
