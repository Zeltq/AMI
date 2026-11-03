"""Resumable P02: preflight, four DAgger rounds, equal-update D0 control.

Run on Windows; native simulation and WSL ROCm training run sequentially.
Completed stages are immutable. Interrupted stages get a new attempt directory;
resume starts from the last completed stage/checkpoint, never a partial dataset.
Create OUTPUT/stop.json to request a graceful stop at the current block/batch.
"""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from ggd_dataset import read_block,decode

ROOT=Path(__file__).resolve().parents[2]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending');temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temporary.replace(path)
def linux(path):
    text=str(path).replace('\\','/')
    return '/mnt/'+text[0].lower()+text[2:] if len(text)>1 and text[1]==':' else text
def score(report):
    sides=[s for g in report['per_match'] for s in g['sides'] if s['controller']=='GGD']
    return (sum(s['first_death_game_seconds'] is not None and s['first_death_game_seconds']<120 for s in sides),-sum(s['lh']+s['deny'] for s in sides))

def compare(teacher,dagger):
    a=read(teacher/'dataset.json');b=read(dagger/'dataset.json');counts={k:[0,0] for k in ('fp16_bits','labels','resets','valid','previous')}
    if len(a['files'])!=len(b['files']):raise ValueError('Parity block counts differ')
    for ea,eb in zip(a['files'],b['files']):
        ha,oa,sa=read_block(teacher/ea['file'],ea['sha256']);hb,ob,sb=read_block(dagger/eb['file'],eb['sha256'])
        for key in ('actors','columns','buckets','horizon'):assert ha[key]==hb[key],key
        pairs={'fp16_bits':(oa.view(np.uint16),ob.view(np.uint16)),'labels':(sa[:,:,:6].view(np.uint32),sb[:,:,:6].view(np.uint32)),
               'resets':(sa[:,:,10],sb[:,:,10]),'valid':(sa[:,:,11],sb[:,:,11]),'previous':(decode(ha,oa,sa)['previous'],decode(hb,ob,sb)['previous'])}
        for key,(x,y) in pairs.items():counts[key][0]+=int((x==y).sum());counts[key][1]+=x.size
    result={'passed':all(n==d for n,d in counts.values()),'blocks':len(a['files']),'compared':counts,'matches':a['config']['games'],
            'scope':'all actors and padded frames; labels compare float32 bits; source/DAgger flags intentionally differ'}
    if not result['passed']:raise ValueError('beta=1 parity failed: '+json.dumps(result))
    return result

def aggregate(datasets,out,round_number):
    entries=[];source_manifests=[];base=None
    for index,root in enumerate(datasets):
        root=Path(root);manifest=read(root/'dataset.json')
        if manifest['partial']:raise ValueError('Cannot aggregate a partial dataset')
        if base is None:base=manifest
        source='teacher-demo' if index==0 else f'dagger-round-{index}'
        source_manifests.append({'path':str(root/'dataset.json'),'sha256':hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest(),'source':source})
        entries.extend(dict(e,file=linux((root/e['file']).resolve()),source=source) for e in manifest['files'])
    result={k:base[k] for k in ('family','observation_profile','teacher_sha256')}
    result.update(format_version=3,partial=False,files=entries,sources=source_manifests,new_source=f'dagger-round-{round_number}')
    write(out/'dataset.json',result)

class Runner:
    def __init__(self,config,out):
        self.cfg=config;self.out=out;self.out.mkdir(parents=True,exist_ok=True)
        self.state=read(out/'state.json') if (out/'state.json').exists() else {'config':config,'stages':{},'jobs':[]}
        if self.state['config']!=config:raise ValueError('Resume configuration differs')
        self.reports=ROOT/'reports/P02';self.reports.mkdir(exist_ok=True)
    def save(self):write(self.out/'state.json',self.state)
    def status(self,stage,**kwargs):
        value={'stage':stage,'driver_pid':os.getpid(),'run':str(self.out),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs}
        write(self.out/'status.json',value)
        # Preserve other agents' sections. This is a short read/replace, not a
        # monitor claiming or launching another agent's job.
        shared=Path('C:/AMI/runs/pilot-status.json');all_status=read(shared) if shared.exists() else {};all_status['P02']=value;write(shared,all_status)
    def job(self,name,command,prepare=None):
        if name in self.state['stages']:return Path(self.state['stages'][name]['output'])
        if (self.out/'stop.json').exists():raise InterruptedError('stop.json requested')
        attempt=1
        while (self.out/f'{name}-attempt-{attempt}').exists():attempt+=1
        target=self.out/f'{name}-attempt-{attempt}';target.mkdir()
        if prepare:prepare(target)
        args=command(target);started=time.monotonic()
        job={'stage':name,'argv':args,'command':subprocess.list2cmdline(args),'output':str(target),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with (target/'process.log').open('w') as log:
            process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);job['pid']=process.pid
            self.state['jobs'].append(job);self.save();self.status(**job)
            while process.poll() is None:
                if (self.out/'stop.json').exists():write(target/'stop.json',{'requested_by':'run_dagger'})
                time.sleep(5)
            job.update(exit_code=process.returncode,minutes=(time.monotonic()-started)/60);self.save()
        if process.returncode:raise RuntimeError(f'{name}: exit {process.returncode}; see {target}/process.log')
        if (self.out/'stop.json').exists():raise InterruptedError('stop.json requested')
        result=read(target/('result.json' if name.startswith('train-') else 'report.json'))
        if result.get('partial') or result.get('complete') is False and 'smoke' not in name:raise RuntimeError('Incomplete stage: '+name)
        self.state['stages'][name]={'output':str(target),'minutes':job['minutes']};self.save()
        self.archive(name,result)
        return target
    def archive(self,name,result):
        if 'per_match' in result:
            keys=('side','controller','lh','deny','deaths','first_death_game_seconds','mid_arrival_game_seconds','mid_arrival_elapsed_seconds','attack_cancellations','actions')
            result={**result,'per_match':[dict(game=g['game'],lane_deaths=g['lane_deaths'],game_seconds=g['game_seconds'],sides=[{k:s[k] for k in keys} for s in g['sides']]) for g in result['per_match']]}
        write(self.reports/f'{name}.json',result)
        write(self.reports/'jobs.json',self.state['jobs'])
        for p in self.reports.glob('*.json'):
            if p.stat().st_size>=1_000_000:raise ValueError('Report too large: '+str(p))
    def collect(self,name,weights,beta,games,parallel,round_number,teacher=False):
        cfg={'engine':self.cfg['engine'],'games':games,'parallel_matches':parallel,'seconds':self.cfg['seconds'],'seed':self.cfg['collection_seed']+round_number*100003}
        if not teacher:cfg={'collection':cfg,'weights':str(weights),'beta':beta,'round':max(1,round_number)}
        return self.job(name,lambda t:[self.cfg['executable'],'collect-teacher' if teacher else 'collect-dagger',str(t/'input.json'),str(t)],lambda t:write(t/'input.json',cfg))
    def evaluate(self,name,weights,matches=64,selection='sampling',teacher=False,trace=False):
        args=[self.cfg['executable'],'evaluate-ggd',self.cfg['engine'],str(weights),'OUTPUT','--mode','teacher','--matches',str(matches),'--seconds',str(self.cfg['seconds']),'--seed',str(self.cfg['evaluation_seed']),'--'+selection]
        if teacher:args.append('--record-teacher')
        if trace:args.append('--trace')
        return self.job(name,lambda t:[str(t) if x=='OUTPUT' else x for x in args])
    def train(self,name,dataset,checkpoint,updates=None,dagger=True,smoke=False):
        cfg={'dataset':str(dataset),'epochs':1 if dagger else 100,'batch_sequences':8,'learning_rate':1e-4,'seed':20261102,'gradient_norm':1.,'attack_weight':4.,'dagger':dagger}
        args=['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh'),linux(ROOT/'gpu/scripts/train_ggd.py'),'CONFIG','OUTPUT','--initialize',linux(checkpoint)]
        if updates:args+=['--max-updates' if smoke else '--updates',str(updates)]
        return self.job(name,lambda t:[linux(t/'input.json') if x=='CONFIG' else linux(t) if x=='OUTPUT' else x for x in args],lambda t:write(t/'input.json',cfg))
    def export(self,trained):
        # Reuse the stable weight exporter. Record command alongside other jobs,
        # but duration reporting is reserved for collection/training/evaluation.
        weights=trained/'export/weights'
        if not (weights/'manifest.json').exists():
            args=['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh'),linux(ROOT/'gpu/scripts/weights.py'),'--checkpoint',linux(trained/'last.pt'),'--output',linux(trained/'export')]
            self.status('export',command=subprocess.list2cmdline(args),artifact=str(weights))
            subprocess.run(args,cwd=ROOT,check=True)
        return weights
    def preflight(self):
        weights=self.cfg['initial_weights']
        teacher=self.collect('parity-teacher',weights,1.,2,2,0,teacher=True)
        dagger=self.collect('parity-dagger',weights,1.,2,2,0)
        write(self.reports/'parity.json',compare(teacher,dagger))
        collected=self.collect('smoke-collect',weights,.5,8,8,1)
        aggregate([self.cfg['dataset'],collected],self.out/'smoke-aggregate',1)
        trained=self.train('train-smoke',self.out/'smoke-aggregate',self.cfg['initial_checkpoint'],50,smoke=True)
        weights=self.export(trained)
        for selection in ('sampling','argmax'):self.evaluate('smoke-'+selection,weights,8,selection)
        write(self.out/'preflight.json',{'passed':True,'updates':read(trained/'result.json')['updates']})
    def full(self):
        if not (self.out/'preflight.json').exists():raise ValueError('Run preflight before full experiment')
        datasets=[self.cfg['dataset']];checkpoint=Path(self.cfg['initial_checkpoint']);weights=Path(self.cfg['initial_weights']);total_updates=0;candidates=[]
        self.evaluate('teacher-baseline',weights,64,teacher=True)
        for selection in ('sampling','argmax'):self.evaluate('M0-'+selection,weights,64,selection)
        for k,beta in enumerate((.5,.25,.1,0.),1):
            collected=self.collect(f'round-{k}-collect',weights,beta,256,128,k);datasets.append(str(collected))
            aggregated=self.out/f'aggregate-{k}';aggregate(datasets,aggregated,k)
            trained=self.train(f'train-round-{k}',aggregated,checkpoint);total_updates+=read(trained/'result.json')['updates']
            checkpoint=trained/'last.pt';weights=self.export(trained)
            for selection in ('sampling','argmax'):
                evaluation=self.evaluate(f'round-{k}-{selection}',weights,64,selection)
                if selection=='sampling':candidates.append({'round':k,'score':score(read(evaluation/'report.json')),'weights':str(weights),'checkpoint':str(checkpoint)})
            self.state.update(total_updates=total_updates,candidates=candidates,best=min(candidates,key=lambda x:x['score']));self.save()
        control=self.train('train-control',Path(self.cfg['dataset']),Path(self.cfg['initial_checkpoint']),total_updates,dagger=False)
        control_weights=self.export(control)
        for selection in ('sampling','argmax'):self.evaluate('control-'+selection,control_weights,64,selection)
        self.evaluate('best-traces',self.state['best']['weights'],4,'sampling',trace=True)
        write(self.reports/'selection.json',{'criterion':'sampling: first fewer deaths before game time 120s, then greater own LH+deny','best':self.state['best'],'candidates':candidates,'control_updates':total_updates})

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');p.add_argument('--phase',choices=['preflight','full','all'],default='all');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    # OS lock is released even after a driver crash; a second driver cannot
    # launch overlapping jobs. Existing live orphan children must be handled.
    import msvcrt
    with (out/'driver.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        runner=Runner(read(a.config),out)
        for job in runner.state['jobs']:
            if 'exit_code' not in job:
                probe=subprocess.run(['tasklist','/FI',f"PID eq {job['pid']}",'/FO','CSV','/NH'],capture_output=True,text=True)
                if f'"{job["pid"]}"' in probe.stdout:raise RuntimeError(f"Previous child PID {job['pid']} still alive; refusing duplicate")
        try:
            if a.phase in ('preflight','all'):runner.preflight()
            if a.phase in ('full','all'):runner.full()
            runner.status('complete-'+a.phase)
        except BaseException as error:
            runner.status('stopped' if isinstance(error,InterruptedError) else 'failed',error=str(error));raise

if __name__=='__main__':main()
