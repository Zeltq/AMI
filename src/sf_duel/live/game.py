"""Scoped dedicated-server process and reversible bot-script installation."""
from pathlib import Path
import json
import subprocess
import sys
import time
import re
from .console import write_console_command

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_GAME=Path('C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta')

def duel_clients(log):
    """Read engine client userids by our unique names, including SourceTV offsets."""
    result={}
    for userid,team,index in re.findall(r"(?m)^\s*(\d+)\s+BOT[^\n]*'SF_DUEL_([23])_([1-5])(?: [^']*)?'\s*$",log):
        result[(int(team),int(index))]=int(userid)
    return result

class Game:
    def __init__(self,run,match='probe',timescale=1.,hz=15,game=DEFAULT_GAME,game_mode=21,game_port=27155,diagnostic_bot_level=None,bot_root=None,recording=None,collect_teacher=False,content_root=None,engine_threads=None,worldstate_ports=None,worldstate_frames=2,worldstate_threaded=True,developer_mode=False,hero_seed=None,console_debug=False):
        self.run=Path(run);self.match=match;self.timescale=timescale;self.hz=hz
        self.game=Path(game);self.process=None;self.pid=None;self.installed=False
        self.game_mode=game_mode;self.game_port=game_port
        self.diagnostic_bot_level=diagnostic_bot_level
        self.bot_root=Path(bot_root) if bot_root else ROOT/'bots'
        self.recording=recording
        self.collect_teacher=collect_teacher
        self.engine_threads=engine_threads
        self.developer_mode=developer_mode
        self.hero_seed=hero_seed
        self.console_debug=console_debug
        self.worldstate_ports=worldstate_ports
        if worldstate_frames not in (1,2):raise ValueError('Native observation period must be 1 or 2 ticks')
        self.worldstate_frames=worldstate_frames
        self.worldstate_threaded=worldstate_threaded
        self.content_root=Path(content_root) if content_root else self.game/'game/dota'
        self.link=self.content_root/'scripts/vscripts/bots'
        self.backup=self.link.with_name('bots.sf_duel_backup')
        self.log=self.run/'dota.log'

    def command(self,text):
        if self.pid is None: raise RuntimeError('No dedicated-server PID')
        write_console_command(self.pid,text)

    def wait_log(self,needle,timeout=100):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.log.exists() and needle in self.log.read_text(errors='replace'):
                return
            if self.process.poll() is not None:
                raise RuntimeError(f'Dota ended before {needle}; inspect {self.log}')
            time.sleep(.1)
        raise TimeoutError(f'Dota did not report {needle}; inspect {self.log}')

    def __enter__(self):
        if sys.platform!='win32': raise RuntimeError('Launch Windows Dota from Windows Python')
        self.run.mkdir(parents=True,exist_ok=True)
        if self.backup.exists(): raise RuntimeError(f'Unrestored bot backup exists: {self.backup}')
        if self.link.exists():
            if not self.link.is_junction():
                raise RuntimeError('Existing bots directory is not a junction; refusing to move it')
            self.link.rename(self.backup)
        try:
            def ps(s): return "'"+str(s).replace("'","''")+"'"
            subprocess.run(['powershell','-NoProfile','-Command',
                f'New-Item -ItemType Junction -Path {ps(self.link)} -Target {ps(self.bot_root)} | Out-Null'],
                check=True,capture_output=True)
            self.installed=True
            controller='policy' if self.bot_root==ROOT/'bots' else 'teacher'
            settings=f'return {{ endpoint="http://127.0.0.1:12260/act", interval={1/self.hz}, match="{self.match}", collect_teacher={str(self.collect_teacher).lower()}, controller_type="{controller}" }}\n'
            (self.bot_root/'settings.lua').write_text(settings,encoding='utf-8')
            for player in range(10):
                (self.bot_root/f'response_{self.match}_{player}.lua').write_text('return {seq=0,action=0,buy=0}\n')
            exe=self.game/'game/bin/win64/dota2.exe'
            args=[str(exe),'-dedicated','-insecure','-ip','127.0.0.1','-port',str(self.game_port),
                '+map','start','gamemode',str(self.game_mode),'+sv_lan','1','+sv_cheats','1','-fill_with_bots',
                '+sv_hibernate_when_empty','0','+dota_auto_surrender_all_disconnected_timeout','100000',
                '+host_timescale',str(self.timescale)]
            if self.content_root!=self.game/'game/dota':
                args[1:1]=['-game',str(self.content_root)]
            if self.engine_threads is not None:args[1:1]=['-threads',str(self.engine_threads)]
            if self.developer_mode:args.append('-dev')
            if self.console_debug:
                if self.content_root==DEFAULT_GAME/'game/dota':raise ValueError('Console capture requires isolated content')
                # isolated_content may have hardlinked the old Steam console log.
                (self.content_root/'console.log').unlink(missing_ok=True)
                args.append('-condebug')
            if self.worldstate_ports:
                args+=['-botworldstatetosocket_radiant',str(self.worldstate_ports[0]),
                    '-botworldstatetosocket_dire',str(self.worldstate_ports[1]),
                    '-botworldstatetosocket_frames',str(self.worldstate_frames)]
                if self.worldstate_threaded:args.append('-botworldstatesocket_threaded')
            if self.diagnostic_bot_level is not None:args+=['+dota_bot_level',str(self.diagnostic_bot_level)]
            if self.recording:
                i=args.index('+map')
                args[i:i]=['+tv_enable','1','+tv_delay','0','+tv_dota_auto_record','0']
            pid_file=self.run/'dota.pid'
            startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow=subprocess.SW_HIDE
            self.process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('console.py')),'launch',
                '--log',str(self.log),'--cwd',str(exe.parent),'--pid-file',str(pid_file),'--',*args],
                creationflags=subprocess.CREATE_NEW_CONSOLE|subprocess.CREATE_NEW_PROCESS_GROUP,
                startupinfo=startup)
            deadline=time.monotonic()+20
            while not pid_file.exists():
                if time.monotonic()>deadline: raise TimeoutError('No Dota PID file')
                time.sleep(.05)
            self.pid=int(pid_file.read_text())
            self.wait_log('Source2Init OK')
            if self.content_root!=DEFAULT_GAME/'game/dota':
                expected=str(self.link).lower().replace('/','\\')
                deadline=time.monotonic()+30
                while True:
                    paths=re.findall(r'bot scripting using script path ([^\r\n]+)',self.log.read_text(errors='replace'))
                    if paths:
                        if any(p.lower().replace('/','\\')!=expected for p in paths):
                            raise RuntimeError(f'Bot script isolation failed: {paths}')
                        break
                    if time.monotonic()>deadline:raise TimeoutError('No isolated bot script path reported')
                    time.sleep(.1)
            deadline=time.monotonic()+100
            while True:
                text=self.log.read_text(errors='replace')
                heroes=set(re.findall(r'PR:SetSelectedHero\s+(\d+):[^\n]*npc_dota_hero_nevermore',text))
                if len(heroes)==2:
                    if self.hero_seed is not None:
                        for player in sorted(heroes):self.command(f'dota_pick_hero {player} npc_dota_hero_nevermore {int(self.hero_seed)}')
                    break
                if time.monotonic()>deadline:raise TimeoutError('Both SF heroes were not selected')
                time.sleep(.1)
            self.command('dota_local_map_strategy_time 0')
            self.command('dota_tutorial_skip_pregame 1')
            self.wait_log('DOTA_GAMERULES_STATE_PRE_GAME')
            # SourceTV has its own client userid. Player IDs are NOT kick userids.
            self.command('status')
            deadline=time.monotonic()+10
            while True:
                clients=duel_clients(self.log.read_text(errors='replace'))
                if (2,1) in clients and (3,1) in clients:break
                if time.monotonic()>deadline:raise RuntimeError('Could not map named bots to engine client userids')
                time.sleep(.05)
            extras=[userid for (team,index),userid in clients.items() if index!=1]
            for userid in sorted(extras,reverse=True):self.command(f'kickid {userid}')
            if self.bot_root!=ROOT/'bots':self.command('ent_remove_all npc_dota_hero_lina')
            if self.recording:self.command(f'tv_record {self.recording}')
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self,*exc):
        if self.pid is not None:
            try:
                if self.recording:self.command('tv_stoprecord')
                self.command('quit')
                self.process.wait(timeout=8)
            except (OSError,RuntimeError,subprocess.TimeoutExpired):
                pass
            found=subprocess.run(['tasklist','/FI',f'PID eq {self.pid}','/FO','CSV'],capture_output=True,text=True)
            if 'dota2.exe' in found.stdout.lower():
                subprocess.run(['taskkill','/PID',str(self.pid),'/T','/F'],capture_output=True,check=True)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate();self.process.wait(timeout=5)
        if self.installed:
            if not self.link.is_junction() or self.link.resolve()!=self.bot_root.resolve():
                raise RuntimeError('Bot junction changed during run; refusing to remove it')
            self.link.rmdir()
        if self.backup.exists(): self.backup.rename(self.link)
        for player in range(10):
            response=self.bot_root/f'response_{self.match}_{player}.lua'
            response.unlink(missing_ok=True)
