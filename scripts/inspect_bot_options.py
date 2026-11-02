"""Read local engine option names; does not modify binaries."""
import json
import mmap
from pathlib import Path

path=Path('C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota/bin/win64/server.dll')
found=set()
with path.open('rb') as source,mmap.mmap(source.fileno(),0,access=mmap.ACCESS_READ) as data:
    for needle in (b'bot_script',b'creeps_no',b'spawn_neutral',b'spawn_creeps'):
        position=0
        while (index:=data.find(needle,position))>=0:
            start=data.rfind(b'\x00',max(0,index-120),index)+1
            end=data.find(b'\x00',index)
            value=data[start:end].decode('ascii',errors='replace')
            if len(value)<250:found.add(value)
            position=index+len(needle)
print(json.dumps(sorted(found),indent=2))
