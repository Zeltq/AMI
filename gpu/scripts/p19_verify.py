"""Verify saved P19 balances, complete coverage, and the paid-start smoke match."""
import json
from pathlib import Path

ROOT=Path('C:/AMI/runs/p19')


def main():
    checks={}
    for name in ['baseline-chio-v9-r4','baseline-v9-v9']:
        run=ROOT/name;data=json.loads((run/'analysis.json').read_text())
        awards=json.loads((run/'creep-awards.json').read_text())
        assert data['attributed']==data['credits']
        for team,side in data['teams'].items():
            assert side['samples']==70 and not side['missing_times']
            assert side['initial']['gold']==600
            assert side['passive_observed_gold']==900
            assert [m['gold'] for m in side['passive_intervals']]==[90]*10
            assert side['final']['deaths']==0
            spent=sum(p.get('gold_before',p['gold'])-p['gold'] for p in side['purchases'])
            bounty=sum(r['gold'] for r in awards if r['team']==int(team))
            assert side['final']['gold']==600+900+bounty-spent
        assert sum(r['xp'] for r in data['xp_by_type'].values())==sum(s['final']['xp'] for s in data['teams'].values())
        checks[name]={'samples':140,'expected_samples':140,'gold_balances_verified':2,'xp_total_verified':True}
    run=ROOT/'fixed-chio-v9'
    data=json.loads((run/'analysis.json').read_text())
    policy=json.loads((run/'policy/status.json').read_text())
    assert policy['executor_version']=='p17-v3' and not policy['errors']
    for team,side in data['teams'].items():
        band=[p for p in side['purchases'] if p['kind']=='start_purchase']
        assert len(band)==1 and band[0]['success'] and band[0]['gold_before']==600 and band[0]['gold']==95
        tango=next(p for p in side['purchases'] if p['item']=='item_tango' and p['success'])
        assert tango['gold_before']==95 and tango['gold']==5
        assert side['passive_intervals'][0]['gold']==90
        assert not any(p['time']<0 and p['item'] in ('item_boots','item_bracer') for p in side['purchases'])
    rows=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    for r in rows:
        if r['kind'] in ('state','sample'):
            assert sum(i['name']=='item_wraith_band' for i in r['inventory'])<=1
    probes=[r for r in rows if r['kind']=='controlled_siege']
    assert len(probes)==4
    for p in probes:
        assert p['creep_dead']
        before,after=p['before'],p['after']
        if p['denied']:
            assert after['deny']==before['deny']+1 and after['gold']==before['gold']
            assert p['other_after']['xp']-p['other_before']['xp']==44
        else:
            assert after['lh']==before['lh']+1 and 59<=after['gold']-before['gold']<=72
            assert after['xp']-before['xp']==88
    checks['fixed']={'paid_start_verified':2,'heroes':2,'siege_transactions_verified':4,'siege_transactions':4,
                     'executor_version':policy['executor_version'],'policy_errors':len(policy['errors'])}
    (ROOT/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
