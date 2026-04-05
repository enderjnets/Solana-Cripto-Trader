"""
FIX 3.1: Safe I/O utilities for atomic JSON reads/writes.
Prevents state file corruption from crashes during writes.
"""
import json
import os
import tempfile
import logging
from pathlib import Path

log = logging.getLogger("safe_io")


def atomic_write_json(filepath, data: dict, indent: int = 2):
    """Write JSON atomically: write to temp file, then rename.

    This ensures that a crash during write cannot corrupt the original file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        os.rename(tmp_path, str(filepath))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_read_json(filepath, default=None):
    """Read JSON with fallback on corruption or missing file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        return default if default is not None else {}
    except json.JSONDecodeError as e:
        log.error(f"JSON decode error in {filepath}: {e}")
        return default if default is not None else {}
    except Exception as e:
        log.error(f"Error reading {filepath}: {e}")
        return default if default is not None else {}
