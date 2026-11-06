import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'gpu/scripts'))
from p17_summary import summarize


def test_death_event_is_counted_before_next_periodic_snapshot(tmp_path):
    lines=[]
    for team in (2,3):
        lines.append('SF_CHIO_METRIC|'+json.dumps(dict(team=team,time=10,tick=300,x=0,y=0,alive=True,lh=2,deny=1,deaths=0)))
    lines.append('SF_PLAY_EVENT|'+json.dumps(dict(kind='death',team=2,time=10.1)))
    (tmp_path/'dota.log').write_text('\n'.join(lines))
    result=summarize(tmp_path)
    assert result['sides']['2']['deaths']==1
    assert result['sides']['3']['deaths']==0


def test_mid_time_uses_exact_handoff_before_rounded_metric(tmp_path):
    metric=dict(team=2,time=-64,tick=2178,x=-1300,y=-1300,alive=True,lh=0,deny=0,deaths=0)
    handoff=dict(team=2,time=-64.566666,tick=2161,handoffs=1)
    ack=dict(team=2,accepted=True,observed_tick=2154,tick=2161)
    (tmp_path/'dota.log').write_text('\n'.join(['SF_CHIO_METRIC|'+json.dumps(metric),
        'SF_CHIO_HANDOFF|'+json.dumps(handoff),'SF_CHIO_NAV_ACK|'+json.dumps(ack)]))
    side=summarize(tmp_path)['sides']['2']
    assert side['mid_time']==handoff['time']
    assert side['navigation']['orders']==side['navigation']['accepted']==side['navigation']['delay7']==1
