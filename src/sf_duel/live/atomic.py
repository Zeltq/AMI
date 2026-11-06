"""Bounded retries for a Lua reader briefly holding a Windows file handle."""
import time


def replace_text(path,text,timeout=.05):
    temporary=path.with_suffix('.pending')
    temporary.write_text(text,encoding='utf-8')
    deadline=time.monotonic()+timeout;retries=0
    while True:
        try:temporary.replace(path);return retries
        except PermissionError:
            if time.monotonic()>=deadline:raise
            retries+=1;time.sleep(.001)
