"""Reviewable verification of phase coverage and before/after critic diagnostics."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import p08_results as results
ROOT=results.ROOT
OUT=results.OUT/'revised'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def history(name):return [json.loads(x)for x in(ROOT/name/'history.jsonl').read_text(encoding='utf-8').splitlines()]
def save(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

if __name__=='__main__':
    stage=sys.argv[1]if len(sys.argv)>1 else'rev-desync-check'
    before=history('rev-critic-2');after=history(stage);run=read(ROOT/stage/'result.json')
    results.OUT=OUT;results.verify(stage);weights=read(OUT/f'{stage}-verification.json')
    rows=[dict(block=h['block'],decisions=h['learning']['valid_decisions'],phase_counts=h['phase_decisions'],phase_fractions=[v/h['learning']['valid_decisions']for v in h['phase_decisions']],native_clock_verified=h['phase_histogram_verified'],terminal_matches=h['finished_in_batch'],value_loss=h['learning']['value_loss'],ev_post0=h['learning']['ev_post0'],ev_pregame=h['learning']['ev_pregame'])for h in after]
    coverage=len(rows)>=3 and all(r['native_clock_verified']and all(v>0 for v in r['phase_counts'])for r in rows)
    prior_finished=0;prior_batch=-1;sync=[]
    for h in before:
        endings=h['finished_in_batch']-(prior_finished if h['batch']==prior_batch else 0)
        if endings:sync.append(dict(block=h['block'],new_terminal_matches=endings,value_loss=h['learning']['value_loss']))
        prior_finished=h['finished_in_batch'];prior_batch=h['batch']
    loss_peak_before=max(h['learning']['value_loss']for h in before);loss_peak_after=max(h['learning']['value_loss']for h in after)
    passed=coverage and weights['passed'] and sum(r['terminal_matches']for r in rows)>0 and loss_peak_after<loss_peak_before
    report=dict(passed=passed,phase_order=['pregame','0-3min','3-6min','6-10min'],blocks=rows,first_three_consecutive=rows[:3],weights_unchanged=weights['identical'],weights_total=weights['total'],synchronous_terminal_blocks=sync,peak_value_loss_before=loss_peak_before,peak_value_loss_after=loss_peak_after,desynchronization=run['desynchronization'],limitations=['Six-block frozen-weight check; long PPO stability is assessed by the two subsequent trials.','The before curve includes critic learning; after uses its final checkpoint and a different phase distribution. EV values do not represent identical populations.'])
    save(OUT/'desync-validation.json',report)
    offset=before[-1]['training_seconds']/60+run['desynchronization']['seconds']/60
    fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    for ax,key,title in [(axes[0],'ev_post0','EV after 0:00'),(axes[1],'value_loss','Value loss')]:
        ax.plot([h['training_seconds']/60 for h in before],[h['learning'][key]for h in before],label='Synchronous critic warmup')
        ax.plot([offset+h['training_seconds']/60 for h in after],[h['learning'][key]for h in after],'.-',label='Mixed phases, frozen weights')
        ax.axvspan(before[-1]['training_seconds']/60,offset,color='gray',alpha=.15,label='Phase initialization, no updates')
        ax.set_title(title,loc='left');ax.set_xlabel('Accumulated stage time, minutes');ax.grid(alpha=.2);ax.legend(fontsize=8)
    bottoms=[0.]*len(rows)
    for i,label in enumerate(report['phase_order']):
        values=[r['phase_fractions'][i]for r in rows];axes[2].bar([r['block']for r in rows],values,bottom=bottoms,label=label);bottoms=[a+b for a,b in zip(bottoms,values)]
    axes[2].set_title('Decision phase distribution after initialization',loc='left');axes[2].set_xlabel('Consecutive rollout block');axes[2].set_ylabel('Fraction of valid learner decisions');axes[2].legend(ncol=4,fontsize=8)
    path=OUT/'desync-before-after.svg';fig.savefig(path);path.write_bytes(('\n'.join(x.rstrip()for x in path.read_text(encoding='utf-8').splitlines())+'\n').encode());fig.savefig(ROOT/stage/'desync-preview.png',dpi=100);plt.close(fig)
    print(json.dumps({k:report[k]for k in ['passed','first_three_consecutive','peak_value_loss_before','peak_value_loss_after']}))
    if not passed:raise SystemExit('Desynchronization validation needs investigation')
