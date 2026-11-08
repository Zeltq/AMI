import json
from pathlib import Path
import sys
from unittest.mock import patch
from lupa.lua51 import LuaRuntime
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'gpu/scripts'),str(ROOT/'scripts')]
import p17_supervisor as supervisor
from p17_matches import bundle
from sf_duel.chio_version import EXECUTOR_VERSION


def test_bundle_records_v2_for_both_lua_controllers(tmp_path):
    dest=tmp_path/'bots';bundle(dest,2)
    settings=LuaRuntime().execute((dest/'p17_settings.lua').read_text())
    assert settings.executor_version==EXECUTOR_VERSION=='p17-v3'
    assert 'SF_CHIO_EXECUTOR_VERSION' in (dest/'bot_generic.lua').read_text()


def test_v1_results_not_replayed_or_mixed_into_v2(tmp_path):
    def weights(name):
        p=tmp_path/name;p.mkdir();(p/'manifest.json').write_text(json.dumps({'tensors':{'x':{'shape':[1],'sha256':name}}}));return p
    done,old,new=map(weights,['done','old','new'])
    series={'sampling-dagger-2':dict(source=str(done),fingerprint=supervisor.fingerprint(done),games=[{}]*8),
            'self-play-checkpoint-old':dict(source=str(old),fingerprint=supervisor.fingerprint(old),games=[{}]*2)}
    with patch.object(supervisor,'P16',tmp_path),patch.object(supervisor,'dagger_name',return_value='sampling-dagger-2'),patch.object(supervisor,'candidates',return_value=[('sampling-dagger-2',done),('self-play-checkpoint-new',new)]):
        pending=supervisor.pending_candidates(series)
        assert [p[0] for p in pending]==['self-play-checkpoint-new-p17-v3']
        assert len(series['self-play-checkpoint-old']['games'])==2
        assert supervisor.dagger_count(series)==8
        current='self-play-checkpoint-current-p17-v3'
        series[current]=dict(source=str(old),fingerprint=supervisor.fingerprint(old),games=[{}],executor_version=EXECUTOR_VERSION)
        pending=supervisor.pending_candidates(series)
        assert supervisor.choose_candidate(pending,series)[0]==current


def test_report_trims_old_table_rows_to_150_and_keeps_json_history(tmp_path):
    (tmp_path/'reports/P17').mkdir(parents=True);p=tmp_path/'reports/P17.md'
    p.write_text('\n'.join(['prose']*140+['## Приоритет P16','BLOCKED P17: pending'])+'\n',encoding='utf-8')
    series={f'model{i}':dict(games=[],executor_version=EXECUTOR_VERSION) for i in range(6)}
    with patch.object(supervisor,'ROOT',tmp_path):supervisor.refresh_report(series)
    assert len(p.read_text(encoding='utf-8').splitlines())==150
    assert len(json.loads((tmp_path/'reports/P17/series.json').read_text()))==6
