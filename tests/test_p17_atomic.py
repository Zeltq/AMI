from pathlib import Path
import tempfile
from unittest.mock import patch
from sf_duel.live.atomic import replace_text


def test_transient_windows_reader_lock_retries_then_publishes():
    original=Path.replace;calls=[]
    def locked(path,target):
        calls.append(path)
        if len(calls)<=2:raise PermissionError('reader open')
        return original(path,target)
    with tempfile.TemporaryDirectory() as root:
        path=Path(root)/'commands.lua';path.write_text('return {}')
        with patch.object(Path,'replace',locked):assert replace_text(path,'return {1}')==2
        assert path.read_text()=='return {1}'


def test_permanent_lock_has_deadline_and_keeps_previous_file():
    with tempfile.TemporaryDirectory() as root:
        path=Path(root)/'commands.lua';path.write_text('return {}')
        with patch.object(Path,'replace',side_effect=PermissionError('locked')):
            try:replace_text(path,'return {1}',timeout=0)
            except PermissionError:pass
            else:raise AssertionError('must report permanent lock')
        assert path.read_text()=='return {}'
