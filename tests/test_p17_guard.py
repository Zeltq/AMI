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
