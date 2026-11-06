import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'gpu/scripts'))
import p17_p16_guard as guard


def test_cached_utf8_report_survives_windows_codepage(tmp_path):
    row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pause=False,report='Обучение ↔ 世界')
    (tmp_path/'p16-monitor.json').write_text(json.dumps(row,ensure_ascii=False),encoding='utf-8')
    with patch.object(guard,'RUN',tmp_path):assert guard.check()==row


def test_speed_uses_actual_block_delta_and_elapsed_seconds():
    assert guard.speed(['CHiO block=10 wall_seconds=100','CHiO block=12 wall_seconds=150'])==dict(blocks=2,seconds=50.,rate=.04)


def test_new_training_stage_baseline_pauses_after_ten_percent_drop(tmp_path):
    run=tmp_path/'p17';p16=tmp_path/'p16';work=tmp_path/'work'
    for p in (run/'bridge-001',p16/'jobs/teacher-1024',p16/'imitation',work/'reports'):p.mkdir(parents=True)
    (run/'bridge-001/p16-baseline.json').write_text(json.dumps(dict(stderr_tail=['CHiO block=1 wall_seconds=10','CHiO block=2 wall_seconds=20'])))
    (p16/'jobs/teacher-1024/stderr.log').write_text('CHiO block=1 wall_seconds=10\nCHiO block=2 wall_seconds=20')
    (p16/'state.json').write_text(json.dumps(dict(pid=42,artifacts=str(p16/'imitation'))))
    status=p16/'imitation/status.json';status.write_text(json.dumps(dict(step=100,wall_seconds=100)))
    (work/'reports/P17.md').write_text('# Test\nBLOCKED P17: test\n')
    with patch.object(guard,'RUN',run),patch.object(guard,'P16',p16),patch.object(guard,'__file__',str(work/'gpu/scripts/guard.py')),patch.object(guard,'alive',side_effect=lambda pid:bool(pid)):
        first=guard.check(force=True);assert first['training']['baseline']['rate']==1
        status.write_text(json.dumps(dict(step=150,wall_seconds=200)))
        try:guard.check(force=True)
        except RuntimeError as e:assert '10%' in str(e)
        else:raise AssertionError('P17 must pause')
        saved=json.loads((run/'p16-monitor.json').read_text(encoding='utf-8'))
        assert saved['pause'] and saved['allowed_servers']==0
