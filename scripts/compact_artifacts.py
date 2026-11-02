"""Lossless NTFS compression; paths and file contents remain unchanged."""
import ctypes
from pathlib import Path
import subprocess


def allocated_bytes(path):
    api=ctypes.WinDLL('kernel32',use_last_error=True).GetCompressedFileSizeW
    api.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)];api.restype=ctypes.c_ulong
    high=ctypes.c_ulong();ctypes.set_last_error(0);low=api(str(Path(path).resolve()),ctypes.byref(high))
    if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
    return (high.value<<32)|low


def compact_file(path):
    path=Path(path).resolve();before=allocated_bytes(path)
    result=subprocess.run(['compact.exe','/C','/I','/Q',str(path)],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return {'path':str(path),'logical_bytes':path.stat().st_size,'before_bytes':before,
            'after_bytes':allocated_bytes(path),'returncode':result.returncode}
