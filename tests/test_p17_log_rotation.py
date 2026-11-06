from pathlib import Path
from play_vs_chio import current_console_log,rotated_console_log


def test_follows_announced_lobby_log_even_before_it_exists(tmp_path):
    base=tmp_path/'console.log'
    base.write_text("[EngineServiceManager] Tearing off console log, will continue using suffix '.9078667752'\n")
    assert current_console_log(base)==tmp_path/'console.9078667752.log'
    (tmp_path/'console.9999999999.log').write_text('stale unrelated match')
    assert current_console_log(base)==tmp_path/'console.9078667752.log'


def test_next_lobby_uses_base_name_and_last_complete_announcement():
    text="Tearing off console log, will continue using suffix '.123'\nTearing off console log, will continue using suffix '.456'\n"
    assert rotated_console_log(Path('console.789.log'),text)==Path('console.456.log')
    assert rotated_console_log(Path('dota.log'),text)==Path('dota.log')
    assert rotated_console_log(Path('console.log'),"suffix '../../elsewhere'")==Path('console.log')


def test_attaching_to_existing_rotated_log_skips_old_announcements(tmp_path):
    base=tmp_path/'console.log';rotated=tmp_path/'console.123.log'
    base.write_text("Tearing off console log, will continue using suffix '.123'\n")
    rotated.write_text('current observations\n')
    assert current_console_log(base)==rotated
    assert rotated_console_log(rotated,'current observations')==rotated
    base.write_text("Tearing off console log, will continue using suffix '.456'\n")
    assert current_console_log(base)==tmp_path/'console.456.log'
