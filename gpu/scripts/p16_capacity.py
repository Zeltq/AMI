"""Bounded CHiO collection capacity probe; datasets are diagnostic only."""
import json,subprocess,time
from pathlib import Path
from p11_run import save
ROOT=Path(__file__).resolve().parents[2];RUN=Path('C:/AMI/runs/p16');REPORT=ROOT/'reports/P16'
while True:
 p=RUN/'jobs/preflight-model-pair/status.json'
 if p.exists() and json.loads(p.read_text())['stage'] in ('complete','failed'):break
 time.sleep(2)
rows=json.loads((REPORT/"capacity.json").read_text())["rows"] if (REPORT/"capacity.json").exists() else [];started=time.monotonic()
for count in ([256,320,352] if rows else [192,384,768]):
 name=f'capacity-{count}';out=RUN/name
 cfg=dict(engine=str(RUN/'environment/engine.json'),games=count,parallel_matches=count,seconds=600,seed=20261103,observation_profile='chio-v1',teacher_version='v10',zero_sum=True,telemetry=False,max_blocks=1)
 config=REPORT/'configs'/f'{name}.json';save(config,cfg)
 cmd=['C:/Python314/python.exe','gpu/scripts/p16_job.py',name,'--',str(RUN/'p15-base.exe'),'collect-teacher',str(config),str(out)]
 child=subprocess.Popen(cmd,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW);child.wait()
 status=json.loads((RUN/'jobs'/name/'status.json').read_text());row=dict(matches=count,exit_code=child.returncode,minutes=status['elapsed_minutes'])
 if (out/'report.json').exists():
  result=json.loads((out/'report.json').read_text());row.update(decisions=result['valid_decisions'],seconds=result['wall_seconds'],decisions_per_second=result['valid_decisions']/result['wall_seconds'])
 rows.append(row);save(REPORT/'capacity.json',dict(rows=rows,wall_seconds=time.monotonic()-started))
 if child.returncode:break
 if time.monotonic()-started>180:break
successful=[r for r in rows if r['exit_code']==0]
if not successful:raise RuntimeError('No capacity probe succeeded')
chosen=max(r['matches'] for r in successful)
save(RUN/'capacity-result.json',dict(parallel_matches=chosen,rows=rows,wall_seconds=time.monotonic()-started))
