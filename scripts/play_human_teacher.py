"""Launch the prepared local practice game, remove extra bots, then enable v9."""
import argparse,datetime,json,os,socket,struct,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import duel_clients

class Rcon:
    def __init__(self,port,password):
        self.socket=socket.create_connection(('127.0.0.1',port),timeout=1);self.socket.settimeout(1);self.sequence=10
        self.send(1,3,password)
        for _ in range(3):
            identity,kind,_=self.receive()
            if kind==2:
                if identity!=1:raise RuntimeError('Local RCON authentication rejected')
                return
        raise RuntimeError('No RCON authentication reply')
    def send(self,identity,kind,body):
        data=struct.pack('<ii',identity,kind)+body.encode()+b'\0\0';self.socket.sendall(struct.pack('<i',len(data))+data)
    def exact(self,n):
        data=b''
        while len(data)<n:
            chunk=self.socket.recv(n-len(data))
            if not chunk:raise EOFError('Local RCON closed')
            data+=chunk
        return data
    def receive(self):
        size=struct.unpack('<i',self.exact(4))[0]
        if not 10<=size<=1_000_000:raise ValueError('Invalid RCON packet')
        data=self.exact(size);identity,kind=struct.unpack('<ii',data[:8]);return identity,kind,data[8:-2].decode(errors='replace')
    def command(self,text):
        self.sequence+=1;self.send(self.sequence,2,text);parts=[];self.socket.settimeout(.25)
        try:
            while True:
                identity,_,part=self.receive()
                if identity==self.sequence:parts.append(part)
        except socket.timeout:
            if not parts:raise
        finally:self.socket.settimeout(1)
        return ''.join(parts)
    def close(self):self.socket.close()

def extra_bot_ids(text,human_team):
    return [userid for (team,index),userid in duel_clients(text).items() if team==human_team or index!=1]

def manual_setup(config,log):
    print('Pick SF, wait until the hero appears, then optionally pause with F9.',flush=True)
    start=log.stat().st_size if log.exists() else 0
    input('In the Dota console enter: status. Then press Enter here. ')
    text=log.read_bytes()[start:].decode('utf-8',errors='replace');clients=duel_clients(text)
    if (config['teacher_team'],1) not in clients:raise RuntimeError('Opponent bot missing from status output')
    cfg=Path(config['content'])/'cfg/ami_teacher_cleanup.cfg'
    cfg.write_text('\n'.join(f'kickid {v}' for v in sorted(extra_bot_ids(text,config['human_team']),reverse=True))+'\nent_remove_all npc_dota_hero_lina\n')
    start=log.stat().st_size
    input('In Dota enter: exec ami_teacher_cleanup; status. Then press Enter here. ')
    text=log.read_bytes()[start:].decode('utf-8',errors='replace')
    if (config['teacher_team'],1) not in duel_clients(text) or extra_bot_ids(text,config['human_team']):raise RuntimeError('Status does not confirm exactly one opponent bot')
    if 'SF_V9_HUMAN_READY|' not in log.read_text(errors='replace'):raise RuntimeError('Human SF is not ready on the configured side')
    print('Composition verified. Unpause with F9 if you paused.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('prepared');p.add_argument('--manual',action='store_true');a=p.parse_args();prepared=Path(a.prepared).resolve();config=json.loads((prepared/'match.json').read_text())
    if sys.platform!='win32':raise RuntimeError('The graphical Dota client requires Windows')
    existing=subprocess.check_output(['tasklist','/FI','IMAGENAME eq dota2.exe','/FO','CSV'],text=True)
    if 'dota2.exe' in existing.lower():raise RuntimeError('Close the existing Dota client before launching this isolated match; it has not been stopped')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6];session=prepared/'sessions'/stamp;session.mkdir(parents=True)
    bots=Path(config['bots']);(bots/'human_ready.lua').write_text('return false\n')
    log=Path(config['console_log'])
    if log.exists():log.replace(session/'previous-console.log')
    args=[config['executable'],'-insecure','-console','-novid','-condebug','-conclearlog','-port',str(config['port']),'+exec','ami_teacher_v9.cfg']
    process=subprocess.Popen(args,cwd=Path(config['executable']).parent,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    status={'stage':'waiting_for_local_match','client_pid':process.pid,'human_team':config['human_team'],'teacher_team':config['teacher_team'],'session':str(session)}
    def record(**updates):status.update(updates);(session/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    record();print('Dota launched. Pick Shadow Fiend on '+('Radiant' if config['human_team']==2 else 'Dire')+'.',flush=True)
    connection=None;deadline=time.monotonic()+240
    try:
        if a.manual:manual_setup(config,log)
        while not a.manual and time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Dota exited before the practice match was ready')
            try:
                if connection is None:connection=Rcon(config['port'],config['rcon_password'])
                clients=connection.command('status')
                for userid in sorted(extra_bot_ids(clients,config['human_team']),reverse=True):connection.command(f'kickid {userid}')
                after=connection.command('status');remaining=duel_clients(after)
                bot_ok=(config['teacher_team'],1) in remaining and not extra_bot_ids(after,config['human_team'])
                text=log.read_text(errors='replace') if log.exists() else ''
                if 'SF_V9_ERROR|' in text:raise RuntimeError('Teacher Lua error; inspect console.log')
                if bot_ok and 'SF_V9_HUMAN_READY|' in text:
                    connection.command('ent_remove_all npc_dota_hero_lina');break
            except (OSError,EOFError):
                if connection:connection.close()
                connection=None
            time.sleep(.5)
        else:
            if not a.manual:raise TimeoutError('Local setup timed out. Teacher remains disabled; inspect console.log and select SF on the configured side. Use Play-manual.cmd if local RCON is unavailable.')
        (bots/'human_ready.lua').write_text('return true\n')
        record(stage='playing');print('Ready: one human SF versus teacher-v9. Commands run every 4 ticks with a 7-tick execution delay.',flush=True)
        while process.poll() is None:
            text=log.read_text(errors='replace') if log.exists() else ''
            if 'SF_V9_ERROR|' in text:raise RuntimeError('Teacher Lua error; inspect console.log')
            ends=[line.split('SF_V9_END|',1)[1] for line in text.splitlines() if 'SF_V9_END|' in line]
            if ends:
                result=json.loads(ends[-1]);record(stage='complete',result=result);(session/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);break
            time.sleep(1)
    except BaseException as error:
        record(stage='setup_or_game_failed',error=str(error));raise
    finally:
        if connection:connection.close()
        if log.exists():
            import shutil
            shutil.copy2(log,session/'console.log')
    print('Session artifacts: '+str(session))

if __name__=='__main__':main()
