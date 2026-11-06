import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gpu/scripts'),str(ROOT/'scripts')]
import p17_supervisor as supervisor


def test_active_navigation_pipeline_requires_its_own_dagger_parity(tmp_path):
    stages={'dagger-2-il':{'folder':str(tmp_path/'dagger-2-il')},'parity-dagger-2':{}}
    (tmp_path/'stages.json').write_text(json.dumps(stages))
    with patch.object(supervisor,'P16',tmp_path):
        assert supervisor.candidates()==[] # Original chain is retired even if its parity passed.
        (tmp_path/'navigation-mode.json').write_text(json.dumps({'active':True}))
        assert supervisor.candidates()==[]
        stages['nav-dagger-2-il']={'folder':str(tmp_path/'nav-dagger-2-il')}
        (tmp_path/'stages.json').write_text(json.dumps(stages));assert supervisor.candidates()==[]
        stages['parity-nav-dagger-2']={};(tmp_path/'stages.json').write_text(json.dumps(stages))
        assert supervisor.candidates()==[('nav-dagger-2',tmp_path/'nav-dagger-2-il/weights')]


def test_script_navigation_simulator_result_has_matching_control_mode(tmp_path):
    weights=tmp_path/'weights';weights.mkdir()
    (weights/'manifest.json').write_text(json.dumps({'tensors':{'x':{'shape':[1],'sha256':'abc'}}}))
    (tmp_path/'summary.json').write_text('{}')
    (tmp_path/'evaluations.json').write_text(json.dumps([dict(complete=True,weights=str(weights),summary='summary.json',scripted_base_to_mid=True)]))
    with patch.object(supervisor,'P16',tmp_path),patch.object(supervisor,'P16ROOT',tmp_path):
        result=supervisor.simulator_reports(weights)
    assert result[0]['directly_comparable'] is True
    assert result[0]['dota_base_navigation']=='script'
    assert result[0]['simulator_base_navigation']=='script'


def test_rl_candidates_require_navigation_engine(tmp_path):
    folder=tmp_path/'ppo';weights=folder/'checkpoint-000001/weights';weights.mkdir(parents=True)
    (weights/'manifest.json').write_text('{}')
    (folder/'latest.json').write_text(json.dumps(dict(block=1,path=str(weights.parent))))
    engine=tmp_path/'engine.json';engine.write_text(json.dumps(dict(chio_script_navigation=False)))
    (folder/'config.json').write_text(json.dumps(dict(engine=str(engine))))
    with patch.object(supervisor,'P16',tmp_path):
        assert supervisor.candidates()==[]
        engine.write_text(json.dumps(dict(chio_script_navigation=True)))
        assert supervisor.candidates()==[('ppo-checkpoint-000001',weights)]


def test_report_count_refreshes_after_first_match(tmp_path):
    (tmp_path/'reports/P17').mkdir(parents=True)
    report=tmp_path/'reports/P17.md'
    report.write_text('# Test\nP16 DAgger2:0/8 матчей\n## Приоритет P16\nBLOCKED P17: pending\n',encoding='utf-8')
    series={'nav-dagger-2':{'games':[]}}
    with patch.object(supervisor,'ROOT',tmp_path),patch.object(supervisor,'dagger_name',return_value='nav-dagger-2'):
        supervisor.refresh_report(series)
        series['nav-dagger-2']['games'].append(dict(winner=2,model_team=2,summary={'sides':{'2':dict(lh=1,deny=0,deaths=0,mid_time=10)}}))
        supervisor.refresh_report(series)
    text=report.read_text(encoding='utf-8')
    assert 'nav-dagger-2 1/8' in text and 'nav-dagger-2 0/8' not in text
    assert text.count('<!-- P17_SERIES_BEGIN -->')==1
