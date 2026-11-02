"""Register the v9 bundle as Dota's reversible Local Dev Script installation."""
import argparse,datetime,hashlib,json,subprocess,sys,uuid
from pathlib import Path


def no_dota():
    if sys.platform!='win32':raise RuntimeError('Dota local bot installation requires Windows')
    if 'dota2.exe' in subprocess.check_output(['tasklist','/FI','IMAGENAME eq dota2.exe','/FO','CSV'],text=True).lower():
        raise RuntimeError('Close Dota before changing its local bot scripts; the client has not been stopped')


def junction(link,target):
    def quoted(path):return "'"+str(path).replace("'","''")+"'"
    subprocess.run(['powershell','-NoProfile','-Command','$ErrorActionPreference="Stop"; New-Item -ItemType Junction -Path '+quoted(link)+' -Target '+quoted(target)+' | Out-Null'],check=True,capture_output=True)


def install_configs(prepared,game,saved):
    entries=saved.setdefault('configs',{})
    for source in (prepared/'cfg').glob('*.cfg'):
        target=Path(game)/'game/dota/cfg'/source.name
        if source.name not in entries:
            backup=None
            if target.exists():
                backup=prepared/('previous-'+source.name);backup.write_bytes(target.read_bytes())
            entries[source.name]={'path':str(target),'backup':str(backup) if backup else None}
        elif target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=entries[source.name]['sha256']:
            raise RuntimeError('Installed configuration was changed: '+str(target))
        target.write_bytes(source.read_bytes());entries[source.name]['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    (prepared/'installation.json').write_text(json.dumps(saved,indent=2)+'\n');return saved


def install(prepared):
    prepared=Path(prepared).resolve();config=json.loads((prepared/'lobby.json').read_text());source=Path(config['bots']).resolve()
    for name,expected in json.loads((prepared/'source-manifest.json').read_text()).items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Bot bundle changed: '+name)
    link=Path(config['game'])/'game/dota/scripts/vscripts/bots';record=prepared/'installation.json'
    if record.exists():
        saved=json.loads(record.read_text())
        if saved['installed']:
            if link.is_junction() and link.resolve()==source:return install_configs(prepared,config['game'],saved)
            raise RuntimeError('Installed bot path changed; refusing to replace it')
    no_dota()
    backup=link.with_name('bots.before-ami-teacher-v9-'+uuid.uuid4().hex[:8])
    original=link.exists() or link.is_junction() or link.is_symlink()
    # Rename only the named local bot entry, within the existing vscripts directory.
    if backup.parent.resolve()!=link.parent.resolve() or backup.exists():raise RuntimeError('Invalid backup path')
    previous_target=str(link.resolve()) if original else None
    if original:link.rename(backup)
    try:junction(link,source)
    except BaseException:
        if original:backup.rename(link)
        raise
    saved=dict(installed=True,link=str(link),target=str(source),backup=str(backup) if original else None,previous_target=previous_target,installed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    return install_configs(prepared,config['game'],saved)


def restore(prepared):
    prepared=Path(prepared).resolve();record=prepared/'installation.json';saved=json.loads(record.read_text());no_dota()
    if not saved['installed']:return saved
    link=Path(saved['link']);target=Path(saved['target']).resolve();backup=Path(saved['backup']) if saved['backup'] else None
    if not link.is_junction() or link.resolve()!=target:raise RuntimeError('Local bot path changed; refusing to remove it')
    if backup and (backup.parent.resolve()!=link.parent.resolve() or not (backup.exists() or backup.is_junction())):raise RuntimeError('Original bot backup missing')
    for entry in saved.get('configs',{}).values():
        target=Path(entry['path'])
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Installed configuration changed: '+str(target))
    for entry in saved.get('configs',{}).values():
        target=Path(entry['path'])
        if entry['backup']:target.write_bytes(Path(entry['backup']).read_bytes())
        else:target.unlink(missing_ok=True)
    link.rmdir()  # Remove the junction itself, never its target or bot files.
    if backup:backup.rename(link)
    saved.update(installed=False,restored_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());record.write_text(json.dumps(saved,indent=2)+'\n');return saved


def main():
    p=argparse.ArgumentParser();p.add_argument('prepared');group=p.add_mutually_exclusive_group();group.add_argument('--restore',action='store_true');group.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:no_dota()
    result=restore(a.prepared) if a.restore else install(a.prepared)
    if a.launch:
        config=json.loads((Path(a.prepared)/'lobby.json').read_text());exe=Path(config['game'])/'game/bin/win64/dota2.exe'
        process=subprocess.Popen([str(exe),'-insecure','-console','-novid','-condebug','+dota_bot_practice_start','0'],cwd=exe.parent,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        (Path(a.prepared)/'client.json').write_text(json.dumps({'pid':process.pid,'executable':str(exe),'mode':'normal client; create a local 1v1 lobby'},indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
