"""Apply the user's amended P08 selection rule to immutable evaluation records."""
from pathlib import Path
import p08_revised as runner

RULE = {
    'version': 2,
    'changed_by': 'user, after the first 32-match evaluation',
    'previous': 'early <=1/32, death matches <=4/32; maximize farm; stop if neither eligible',
    'current': 'early <=1/32, death matches <=6/32; maximize farm, but within 5% prefer fewer death matches',
    'relative_gap': '(higher farm - lower farm) / higher farm; strictly less than 0.05',
    'fallback': 'if neither eligible, evaluate both on 64 additional matches, seed 20261104; death matches <=15/96',
    'fallback_early_deaths': 'reported separately; the user specified no new early-death threshold for the pooled 96 matches',
}


def select(rows):
    rows = [dict(r, trial_eligible=r['early_deaths'] <= 1 and r['death_matches'] <= 6)
            for r in rows]
    eligible = [r for r in rows if r['trial_eligible']]
    best = max(eligible, key=lambda r: (r['farm'], -r['death_matches'], -r['learning_rate'])) if eligible else None
    if len(eligible) == 2 and best['farm'] > 0:
        lower = min(eligible, key=lambda r: r['farm'])
        if (best['farm'] - lower['farm']) / best['farm'] < .05:
            best = min(eligible, key=lambda r: (r['death_matches'], -r['farm'], r['learning_rate']))
    return dict(rule=RULE, complete=len(rows) == 2, trials=rows,
                selected=best if len(rows) == 2 else None,
                needs_additional_64=len(rows) == 2 and not eligible,
                stop_required=False)


def refresh():
    rows = []
    for suffix, lr in [('1e5', 1e-5), ('3e5', 3e-5)]:
        name = f'rev-async-eval-{suffix}'
        path = runner.REPORT / f'stage-{name}.json'
        if not path.exists():
            continue
        evaluation = runner.read(path)['evaluation']
        if not evaluation['complete'] or evaluation['matches'] != 32:
            raise ValueError('selection requires a completed 32-match evaluation')
        train = f'rev-async-trial-{suffix}'
        result = runner.read(runner.ROOT / train / 'result.json')
        rows.append(dict(evaluation, learning_rate=lr, train_stage=train,
                         evaluation_stage=name, weights=str(Path(result['latest']) / 'weights')))
    decision = select(rows)
    runner.save(runner.REPORT / 'async-trial-selection-v2.json', decision)
    return decision


if __name__ == '__main__':
    import json
    print(json.dumps(refresh(), ensure_ascii=False))
