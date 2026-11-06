"""Small Win32 boundary: hidden console, kill-on-close job, console commands.

The process supervisor and the short-lived input writer are intentionally
separate processes: attaching a console must never alter the caller's stdio.
"""
import argparse
import ctypes as C
from ctypes import wintypes as W
from pathlib import Path
import subprocess
import sys

class Char(C.Union):
    _fields_=[('unicode',W.WCHAR),('ascii',C.c_char)]
class Key(C.Structure):
    _fields_=[('down',W.BOOL),('repeat',W.WORD),('virtual',W.WORD),
              ('scan',W.WORD),('character',Char),('modifiers',W.DWORD)]
class Event(C.Union):
    _fields_=[('key',Key),('reserved',C.c_byte*16)]
class Input(C.Structure):
    _fields_=[('kind',W.WORD),('event',Event)]
class BasicLimits(C.Structure):
    _fields_=[('process_time',C.c_longlong),('job_time',C.c_longlong),('flags',W.DWORD),
        ('min_working_set',C.c_size_t),('max_working_set',C.c_size_t),('process_count',W.DWORD),
        ('affinity',C.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
class IOCounts(C.Structure):
    _fields_=[(n,C.c_ulonglong) for n in ('reads','writes','other','read_bytes','write_bytes','other_bytes')]
class Limits(C.Structure):
    _fields_=[('basic',BasicLimits),('io',IOCounts),('process_memory',C.c_size_t),
              ('job_memory',C.c_size_t),('peak_process_memory',C.c_size_t),('peak_job_memory',C.c_size_t)]

def kernel():
    k=C.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.restype=W.HANDLE
    k.CreateJobObjectW.restype=W.HANDLE
    k.OpenProcess.restype=W.HANDLE
    k.CloseHandle.argtypes=[W.HANDLE]
    k.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD]
    k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE]
    return k

def send_native(pid,text):
    k=kernel();u=C.WinDLL('user32',use_last_error=True)
    k.FreeConsole()
    if not k.AttachConsole(pid): raise C.WinError(C.get_last_error())
    handle=None
    try:
        handle=k.CreateFileW('CONIN$',0xC0000000,3,None,3,0,None)
        if handle in (None,C.c_void_p(-1).value): raise C.WinError(C.get_last_error())
        chars=text+'\r';events=(Input*len(chars))()
        for event,ch in zip(events,chars):
            mapping=u.VkKeyScanW(ch)
            event.kind=1;event.event.key.down=True;event.event.key.repeat=1
            event.event.key.virtual=13 if ch=='\r' else (mapping&255 if mapping!=-1 else 0)
            event.event.key.scan=u.MapVirtualKeyW(event.event.key.virtual,0)
            event.event.key.character.unicode=ch
            event.event.key.modifiers=0x10 if mapping!=-1 and mapping&0x100 else 0
        written=W.DWORD()
        k.WriteConsoleInputW.argtypes=[W.HANDLE,C.POINTER(Input),W.DWORD,C.POINTER(W.DWORD)]
        if not k.WriteConsoleInputW(handle,events,len(events),C.byref(written)):
            raise C.WinError(C.get_last_error())
        if written.value!=len(events): raise OSError('Partial console command write')
    finally:
        if handle not in (None,C.c_void_p(-1).value): k.CloseHandle(handle)
        k.FreeConsole()

def write_console_command(pid,text):
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'write',str(pid),text],
        capture_output=True,text=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode: raise OSError(result.stderr.strip())

def launch(log,cwd,pid_file,command,below_normal=False):
    k=kernel();job=k.CreateJobObjectW(None,None)
    if not job: raise C.WinError(C.get_last_error())
    limits=Limits();limits.basic.flags=0x2000 # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    process=None
    try:
        if not k.SetInformationJobObject(job,9,C.byref(limits),C.sizeof(limits)):
            raise C.WinError(C.get_last_error())
        with Path(log).open('wb') as output:
            process=subprocess.Popen(command,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if below_normal else 0)
            child=k.OpenProcess(0x100|0x1,False,process.pid) # SET_QUOTA | TERMINATE
            if not child: raise C.WinError(C.get_last_error())
            try:
                if not k.AssignProcessToJobObject(job,child): raise C.WinError(C.get_last_error())
            finally:k.CloseHandle(child)
            pid_path=Path(pid_file);temporary=pid_path.with_suffix('.tmp')
            temporary.write_text(str(process.pid),encoding='ascii');temporary.replace(pid_path)
            return process.wait()
    finally:
        k.CloseHandle(job)
        if process is not None and process.poll() is None: process.wait(timeout=10)

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    writer=sub.add_parser('write');writer.add_argument('pid',type=int);writer.add_argument('text')
    runner=sub.add_parser('launch');runner.add_argument('--log',required=True)
    runner.add_argument('--cwd',required=True);runner.add_argument('--pid-file',required=True)
    runner.add_argument('--below-normal',action='store_true')
    runner.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.mode=='write': send_native(args.pid,args.text)
    else:
        command=args.command[1:] if args.command and args.command[0]=='--' else args.command
        raise SystemExit(launch(args.log,args.cwd,args.pid_file,command,args.below_normal))

if __name__=='__main__':main()
