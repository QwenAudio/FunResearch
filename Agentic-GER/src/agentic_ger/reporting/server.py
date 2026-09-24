#!/usr/bin/env python3
"""Serve a case dashboard with byte ranges for seekable audio playback."""

from __future__ import annotations

import argparse
import io
import math
import os
import shutil
import wave
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, urlsplit


from agentic_ger.utils import REPO_ROOT


ROOT = REPO_ROOT
DEFAULT_DIRECTORY = ROOT / "runs/cases"


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """Add single-range support missing from Python's basic static server."""

    range: tuple[int, int] | None = None
    max_clip_seconds = 600.0

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self) -> BinaryIO | None:
        request = urlsplit(self.path)
        clip_root, marker, clip_path = request.path.partition("/clip/")
        if marker:
            return self._send_clip(clip_root, clip_path, request.query)
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            return super().send_head()
        try:
            source = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        stat = os.fstat(source.fileno())
        size = stat.st_size
        self.range = None
        header = self.headers.get("Range")
        if header:
            try:
                start, end = self._parse_range(header, size)
            except ValueError:
                source.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            self.range = (start, end)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            length = end - start + 1
            source.seek(start)
        else:
            self.send_response(200)
            length = size

        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        return source

    def _send_clip(
        self, clip_root: str, relative_path: str, query: str
    ) -> BinaryIO | None:
        """Return a short PCM WAV excerpt for reliable tunneled playback."""
        if not relative_path.startswith("audio/"):
            self.send_error(404, "Clip source not found")
            return None
        values = parse_qs(query)
        try:
            start = float(values["start"][0])
            end = float(values["end"][0])
        except (KeyError, IndexError, TypeError, ValueError):
            self.send_error(400, "Clip requires numeric start and end")
            return None
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end - start > self.max_clip_seconds
        ):
            self.send_error(400, "Invalid clip interval")
            return None

        source_path = Path(self.translate_path(f"{clip_root}/{relative_path}"))
        try:
            with wave.open(str(source_path), "rb") as reader:
                rate = reader.getframerate()
                start_frame = max(0, round(start * rate))
                end_frame = min(reader.getnframes(), round(end * rate))
                if end_frame <= start_frame:
                    raise ValueError("clip starts beyond the audio")
                reader.setpos(start_frame)
                frames = reader.readframes(end_frame - start_frame)
                parameters = reader.getparams()
        except (OSError, EOFError, ValueError, wave.Error) as error:
            self.send_error(422, f"Cannot cut WAV clip: {error}")
            return None

        clip = io.BytesIO()
        with wave.open(clip, "wb") as writer:
            writer.setparams(parameters)
            writer.writeframes(frames)
        clip.seek(0)
        self.range = None
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(clip.getbuffer().nbytes))
        self.send_header("Accept-Ranges", "none")
        self.end_headers()
        return clip

    @staticmethod
    def _parse_range(header: str, size: int) -> tuple[int, int]:
        if not header.startswith("bytes=") or "," in header or size <= 0:
            raise ValueError("unsupported range")
        start_text, separator, end_text = header[6:].partition("-")
        if not separator:
            raise ValueError("malformed range")
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError("invalid suffix")
            start = max(0, size - suffix)
            end = size - 1
        if start < 0 or start >= size or end < start:
            raise ValueError("range outside file")
        return start, min(end, size - 1)

    def copyfile(self, source: BinaryIO, output: BinaryIO) -> None:
        if self.range is None:
            shutil.copyfileobj(source, output)
            return
        remaining = self.range[1] - self.range[0] + 1
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            output.write(chunk)
            remaining -= len(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12398)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.resolve()
    handler = partial(RangeRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"serving {directory} at http://{args.bind}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
