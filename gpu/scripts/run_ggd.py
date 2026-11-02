"""Run one authorized GGD collection followed by IL; keep process/status artifacts."""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--resume');p.add_argument('--output');a=p.parse_args();run=Path(a.run).resolve()
    out=Path(a.output).resolve() if a.output else run/('GGD0-resumed' if a.resume else 'GGD0')
    config=json.loads((run/'pipeline.json').read_text());exe=Path(config['executable']);status={'family':'GGD','model_name':'GGD0','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':str(run)}
    # Exclusive creation prevents launching the same collection/training twice.
    if a.resume:
        if json.loads((run/'status.json').read_text()).get('stage')!='paused':raise RuntimeError('Resume requires a paused run')
        if sys.platform=='win32' and 'dota2.exe' in subprocess.check_output(['tasklist','/FI','IMAGENAME eq dota2.exe','/FO','CSV'],text=True).lower():raise RuntimeError('Close Dota before resuming GPU training')
        out.mkdir(parents=True,exist_ok=False)
    with ((out/'resume-started.json') if a.resume else (run/'started.json')).open('x') as f:json.dump(status,f)
    def progress(**extra):
        status.update(extra,last_checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(run/'status.json',status)
        pilot=ROOT/'runs/pilot-status.json';saved=json.loads(pilot.read_text(encoding='utf-8'));saved['ggd0_il_20261102']={**status,'background_work_remaining':status.get('stage') not in ('complete','failed','stopped'),'report':str(run/'result.json')};write(pilot,saved)
    def execute(stage,args,folder):
        with (run/f'{stage}.log').open('w') as log:
            process=subprocess.Popen([str(v) for v in args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            progress(stage=stage,command=[str(v) for v in args],child_pid=process.pid,current_status=str(folder/'status.json'))
            while process.poll() is None:
                if (run/'stop.json').exists():
                    (folder/'stop.json').write_text('{}')
                    if stage.endswith('training') and (folder/'status.json').exists():
                        live=json.loads((folder/'status.json').read_text())
                        if live.get('stage')=='gameplay_evaluation':(Path(live['report'].replace('/mnt/c/','C:/')).parent/'stop.json').write_text('{}')
                progress();time.sleep(10)
            if process.returncode:raise RuntimeError(f'{stage} exited {process.returncode}; see {stage}.log')
    try:
        if not a.resume:execute('collection',[exe,'collect-teacher',run/'collect.json',run/'dataset'],run/'dataset')
        if (run/'stop.json').exists():progress(stage='stopped',child_pid=None,command=None);return
        dataset=json.loads((run/'dataset/dataset.json').read_text())
        if dataset['partial'] or len(dataset['game_metrics'])!=config['games']:raise RuntimeError('Collection incomplete; IL was not started')
        if any(g.get('simulation_error',0)!=0 for g in dataset['game_metrics']):raise RuntimeError('Simulator error in demonstrations')
        if (run/'stop.json').exists():progress(stage='stopped',child_pid=None);return
        if a.resume:
            def linux(path):
                s=str(Path(path).resolve()).replace('\\','/');return '/mnt/'+s[0].lower()+s[2:] if len(s)>1 and s[1]==':' else s
            command=['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh'),linux(ROOT/'gpu/scripts/train_ggd.py'),linux(run/'train.json'),linux(out),'--resume',linux(a.resume)]
            execute('resumed-training',command,out)
        else:execute('training',[exe,'train-il',run/'train.json',out],out)
        training=json.loads((out/'result.json').read_text());result={'family':'GGD','model_name':'GGD0','complete':training['complete'],'collection':json.loads((run/'dataset/report.json').read_text()),'teacher':json.loads((run/'dataset/teacher-summary.json').read_text()),'training':training,'epoch_metrics_csv':str(out/'epoch-metrics.csv')}
        write(run/'result.json',result);progress(stage='complete' if result['complete'] else 'stopped',child_pid=None,command=None)
    except Exception:
        write(run/'error.json',{'error':traceback.format_exc(),'status':status});progress(stage='failed',child_pid=None);raise

if __name__=='__main__':main()
