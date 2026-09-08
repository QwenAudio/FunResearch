#!/usr/bin/env python3

import argparse
import ctypes
import os
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf


root = Path(__file__).resolve().parent
system = platform.system()
machine = platform.machine().lower()
if system == "Windows" and machine in ("x86_64", "amd64"):
    library_name = "jaec_x86.dll"
elif system == "Linux" and machine in ("x86_64", "amd64"):
    library_name = "jaec_x86.so"
elif system in ("Darwin", "Linux") and machine in ("aarch64", "arm64"):
    library_name = "jaec_arm.so"
else:
    raise RuntimeError(f"unsupported platform: {system} {platform.machine()}")

library_path = root / "lib" / library_name
if not library_path.is_file():
    raise RuntimeError(f"native library not found: {library_path}")
native = ctypes.CDLL(str(library_path))
pcm_pointer = ctypes.POINTER(ctypes.c_int16)
native.jaec_frontend_create.argtypes = [ctypes.c_char_p]
native.jaec_frontend_create.restype = ctypes.c_void_p
native.jaec_frontend_destroy.argtypes = [ctypes.c_void_p]
native.jaec_frontend_destroy.restype = None
native.jaec_frontend_reset.argtypes = [ctypes.c_void_p]
native.jaec_frontend_reset.restype = None
native.jaec_frontend_process.argtypes = [
    ctypes.c_void_p,
    pcm_pointer,
    pcm_pointer,
    ctypes.c_int,
    pcm_pointer,
]
native.jaec_frontend_process.restype = ctypes.c_int
native.jaec_frontend_last_error.restype = ctypes.c_char_p


def _native_error(fallback):
    error = native.jaec_frontend_last_error()
    return error.decode() if error else fallback


class JAEC:
    def __init__(self, model_bin):
        self._handle = None
        self._handle = native.jaec_frontend_create(os.fsencode(model_bin))
        if not self._handle:
            raise RuntimeError(_native_error("initialization failed"))

    def close(self):
        if self._handle:
            native.jaec_frontend_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def reset(self):
        if not self._handle:
            raise RuntimeError("JAEC instance is closed")
        native.jaec_frontend_reset(self._handle)

    def process(self, mic, ref):
        if not self._handle:
            raise RuntimeError("JAEC instance is closed")
        if not isinstance(mic, np.ndarray) or mic.dtype != np.int16:
            raise TypeError("mic must be a numpy.int16 array")
        if not isinstance(ref, np.ndarray) or ref.dtype != np.int16:
            raise TypeError("ref must be a numpy.int16 array")
        if mic.shape != (160,) or ref.shape != (160,):
            raise ValueError("mic and ref must each contain one 160-sample frame")

        mic = np.ascontiguousarray(mic)
        ref = np.ascontiguousarray(ref)
        output = np.empty(160, dtype=np.int16)
        status = native.jaec_frontend_process(
            self._handle,
            mic.ctypes.data_as(pcm_pointer),
            ref.ctypes.data_as(pcm_pointer),
            160,
            output.ctypes.data_as(pcm_pointer),
        )
        if status != 0:
            raise RuntimeError(_native_error("processing failed"))
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, type=Path)
    parser.add_argument("-o", required=True, type=Path)
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.i, dtype="int16", always_2d=True)
    if sample_rate != 16000 or audio.shape[1] != 2:
        raise ValueError("input must be 16 kHz stereo: ch0=mic, ch1=ref")
    frames = len(audio)
    if frames == 0:
        raise ValueError("input is empty")
    if frames % 160:
        audio = np.pad(audio, ((0, 160 - frames % 160), (0, 0)))

    output = np.empty(len(audio), dtype=np.int16)
    jaec = JAEC(str(root / "weights" / "tde_lp.bin"))
    start_time = time.perf_counter()
    for offset in range(0, len(audio), 160):
        frame = audio[offset : offset + 160]
        output[offset : offset + 160] = jaec.process(frame[:, 0], frame[:, 1])
    elapsed = time.perf_counter() - start_time

    sf.write(args.o, output[:frames], 16000, subtype="PCM_16")
    duration = frames / sample_rate
    print(f"RTF: {elapsed / duration:.4f} ({elapsed:.3f}s / {duration:.3f}s)")


if __name__ == "__main__":
    main()
