"""Read-only Valve VPK v1/v2 access for the installed game's rule data."""
from pathlib import Path
import struct

class Vpk:
    def __init__(self,path):
        self.path=Path(path);self.entries={}
        with self.path.open('rb') as f:
            magic,version,size=struct.unpack('<III',f.read(12))
            if magic!=0x55AA1234 or version not in (1,2):raise ValueError('Unsupported VPK')
            header=12 if version==1 else 28
            f.seek(header);tree=f.read(size)
        self.data_offset=header+size;pos=0
        def string():
            nonlocal pos
            end=tree.index(b'\0',pos);s=tree[pos:end].decode();pos=end+1;return s
        while extension:=string():
            while directory:=string():
                while name:=string():
                    crc,preload,archive,offset,length,terminator=struct.unpack_from('<IHHIIH',tree,pos);pos+=18
                    if terminator!=0xffff:raise ValueError('Invalid VPK entry')
                    prefix=tree[pos:pos+preload];pos+=preload
                    key=('/' if directory==' ' else directory+'/')+name+'.'+extension
                    self.entries[key.lstrip('/')]=(archive,offset,length,prefix)
    def read(self,name):
        archive,offset,length,prefix=self.entries[name]
        path=self.path if archive==0x7fff else self.path.with_name(self.path.name.replace('_dir.vpk',f'_{archive:03}.vpk'))
        with path.open('rb') as f:
            f.seek(offset+(self.data_offset if archive==0x7fff else 0));return prefix+f.read(length)

if __name__=='__main__':
    import sys
    v=Vpk(sys.argv[1])
    if len(sys.argv)>2:
        print(v.read(sys.argv[2]).decode('utf-8-sig'))
    else:
        print('\n'.join(k for k in v.entries if 'scripts/npc/' in k and ('nevermore' in k or k.endswith(('npc_heroes.txt','npc_units.txt','npc_abilities.txt','items.txt')))))
