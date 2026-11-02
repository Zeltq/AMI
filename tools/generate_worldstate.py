"""Generate wire-compatible protobuf bindings from a pinned public interface."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,urllib.request
ROOT=Path(__file__).resolve().parents[1]
repo='https://api.github.com/repos/SteamTracking/Protobufs/commits/master'
commit=json.load(urllib.request.urlopen(repo))['sha']
url=f'https://raw.githubusercontent.com/SteamTracking/Protobufs/{commit}/dota2/dota_gcmessages_common_bot_script.proto'
data=urllib.request.urlopen(url).read()
source=ROOT/'external/worldstate';source.mkdir(parents=True,exist_ok=True)
(source/'upstream.proto').write_bytes(data)
(source/'UPSTREAM.json').write_text(json.dumps({'url':url,'commit':commit,'sha256':hashlib.sha256(data).hexdigest(),
    'adaptation':'Remove non-wire Valve compiler options; preserve field numbers, types and defaults.'},indent=2))
text=data.decode()
text=re.sub(r'(?m)^(?:import|option) [^\n]*\n','',text)
def options(m):
    default=re.search(r'(?:^|,)\s*default\s*=\s*([^,]+)',m[1])
    return '[default = '+default[1]+']' if default else ''
text='syntax = "proto2";\noption go_package = "sfduel/nativecollect/worldstate";\n'+re.sub(r'\[([^\]]*)\]',options,text)
(source/'worldstate.proto').write_text(text)
destination=ROOT/'src/sf_duel/protos';destination.mkdir(exist_ok=True)
(destination/'__init__.py').touch()
subprocess.run([sys.executable,'-m','grpc_tools.protoc','-I'+str(source),
                '--python_out='+str(destination),str(source/'worldstate.proto')],check=True)
print(commit)
