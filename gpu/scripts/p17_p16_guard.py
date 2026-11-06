"""Read-only P16 snapshots; conservative pause of P17 on >10% collection slowdown."""
import ctypes
import datetime
import json
from pathlib import Path
import re

RUN=Path('C:/AMI/runs/p17');P16=Path('C:/AMI/runs/p16')


def alive(pid):
    if not pid:return False
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.restype=ctypes.c_void_p
    k.CloseHandle.argtypes=[ctypes.c_void_p];k.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    handle=k.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    try:
        code=ctypes.c_ulong();return bool(k.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:k.CloseHandle(handle)


def speed(lines):
    points=[]
    for line in lines:
        m=re.search(r'CHiO block=(\d+).*wall_seconds=([\d.]+)',line)
        if m:points.append((int(m[1]),float(m[2])))
    points=points[-8:]
    if len(points)<2 or points[-1][1]<=points[0][1]:return None
    n=points[-1][0]-points[0][0];d=points[-1][1]-points[0][1]
    return dict(blocks=n,seconds=d,rate=n/d)


def check(force=False):
    now=datetime.datetime.now(datetime.timezone.utc);path=RUN/'p16-monitor.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    if not force and previous and (now-datetime.datetime.fromisoformat(previous['utc'])).total_seconds()<1800:
        if previous.get('pause'):raise RuntimeError('P16 slowdown; P17 paused until next read-only check')
        return previous
    baseline=json.loads((RUN/'bridge-001/p16-baseline.json').read_text());base=speed(baseline['stderr_tail'])
    jobs=[]
    for p in (P16/'jobs').glob('*/status.json'):
        s=json.loads(p.read_text())
        if s.get('background_work_remaining'):
            jobs.append(dict(name=p.parent.name,status=s,alive=alive(s.get('child_pid')) or alive(s.get('pid'))))
    current=speed((P16/'jobs/teacher-1024/stderr.log').read_text().splitlines())
    active=any(j['name']=='teacher-1024' and j['alive'] for j in jobs)
    pause=bool(active and base and current and current['rate']<base['rate']*.9)
    report=Path('C:/AMI-wt/P16/reports/P16.md').read_text(encoding='utf-8')
    row=dict(utc=now.isoformat(),baseline=base,current=current,comparable_stage=active,pause=pause,
             allowed_servers=0 if pause else 1,jobs=jobs,report=report,
             note='Block throughput depends on cohort/phase. New training stages have no pre-P17 baseline; report them separately.')
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
    with (RUN/'p16-monitor.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    if pause:
        report_path=Path(__file__).resolve().parents[2]/'reports/P17.md'
        text=report_path.read_text(encoding='utf-8').splitlines()
        text.insert(-1,f"P16 {row['utc']}: {current['blocks']}/{current['seconds']:.3f} блоков/с против {base['blocks']}/{base['seconds']:.3f}; P17 снижена до0/2 серверов (падение>10%).")
        report_path.write_text('\n'.join(text)+'\n',encoding='utf-8')
        raise RuntimeError('P16 throughput fell >10%; pausing all P17 Dota processes')
    return row


if __name__=='__main__':
    row=check(force=True);print(json.dumps({k:v for k,v in row.items() if k not in ('jobs','report')}))
