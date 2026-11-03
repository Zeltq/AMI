"""Two authorized P08 policy trials from one frozen-policy critic checkpoint."""
import argparse,json,os,time
from pathlib import Path
import msvcrt
import p08_revised as runner
import p08_results as results
ROOT,REPORT=runner.ROOT,runner.REPORT

def state(**fields):runner.save(ROOT/'trials-pipeline.json',dict(controller_pid=os.getpid(),**fields))

def main():
    p=argparse.ArgumentParser();p.add_argument('--critic',default='rev-critic-2');args=p.parse_args()
    critic=ROOT/args.critic;deadline=time.monotonic()+900
    state(stage='waiting_for_critic',critic=args.critic)
    while not(critic/'result.json').exists():
        current=runner.read(ROOT/'revised-controller.json')
        if current.get('stage')=='failed':raise RuntimeError('critic controller failed; inspect before trials')
        if time.monotonic()>deadline:raise RuntimeError('critic waiting deadline reached')
        time.sleep(2)
    with(ROOT/'controller.lock').open('r+b')as lock:
        while True:
            try:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:
                if time.monotonic()>deadline:raise RuntimeError('active controller did not release lock')
                time.sleep(2)
        result=runner.read(critic/'result.json');base=runner.read(critic/'config.json')
        if not result['warmup_complete']or result['warmup_post_blocks']<30:
            raise RuntimeError('critic did not complete the prescribed warmup')
        if result['blocks']>150:raise RuntimeError('critic exceeded the block limit')
        for suffix in ['1e5','3e5']:
            if(ROOT/f'rev-trial-{suffix}').exists()or(ROOT/f'rev-eval-{suffix}').exists():
                raise RuntimeError('trial/evaluation already exists; no duplicate runs')
        results.OUT=REPORT;results.verify(args.critic)
        reason='EV threshold'if(result.get('warmup_ev_last5')or -1)>0.3 else'150-block limit'
        runner.save(REPORT/'trial-plan.json',dict(critic=args.critic,critic_checkpoint=result['latest'],warmup_exit_reason=reason,warmup_ev_last5=result.get('warmup_ev_last5'),warmup_post_blocks=result['warmup_post_blocks'],policy_trial_minutes=20,learning_rates=[1e-5,3e-5],epochs=2,selection='early deaths <=1/32 and death matches <=4/32; maximize farm among eligible; stop if neither eligible'))
        for suffix,lr in [('1e5',1e-5),('3e5',3e-5)]:
            cfg=base|dict(minutes=20,max_blocks=0,learning_rate=lr,epochs=2,initialize_weights=str(Path(result['latest'])/'weights'),skip_warmup=True,critic_only=False,checkpoint_minutes=10,evaluate=False)
            runner.save(REPORT/f'trial-{suffix}.json',cfg)
        runner.publish('trial-plan')
        rows=[]
        for suffix,lr in [('1e5',1e-5),('3e5',3e-5)]:
            train=f'rev-trial-{suffix}';state(stage=train,critic=args.critic)
            runner.run_stage(train,[str(runner.EXE),'train-ggd',str(REPORT/f'trial-{suffix}.json'),str(ROOT/train)],1500)
            trained=runner.read(ROOT/train/'result.json');weights=str(Path(trained['latest'])/'weights')
            evaluation=f'rev-eval-{suffix}';state(stage=evaluation,critic=args.critic)
            r=runner.run_stage(evaluation,[str(runner.EXE),'evaluate-ggd',base['engine'],weights,str(ROOT/evaluation),'--mode','teacher','--matches','32','--seconds','600','--seed','20261103','--sampling','--trace'],900)
            rows.append(dict(learning_rate=lr,train_stage=train,evaluation_stage=evaluation,weights=weights,**r['evaluation']))
            runner.save(REPORT/'trial-selection.json',dict(complete=False,trials=rows));runner.publish('trial-selection-progress')
        eligible=[r for r in rows if r['trial_eligible']]
        best=max(eligible,key=lambda r:(r['farm'],-r['death_matches'],-r['early_deaths'],-r['learning_rate']))if eligible else None
        runner.save(REPORT/'trial-selection.json',dict(complete=True,trials=rows,selected=best,stop_required=not eligible))
        runner.publish('trial-selection');state(stage='complete',selected=best,stop_required=not eligible)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        state(stage='failed',error=repr(exc));raise
