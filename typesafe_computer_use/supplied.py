"""Text and addresses the operator handed in. Nothing here is generated or rewritten."""

from __future__ import annotations

import os
from pathlib import Path


class SuppliedError(ValueError):
    """The operator's text or address could not be read as given."""


def supplied_text(text: str | None, path: Path | None) -> str | None:
    """The exact string to type.

    A command-line `--text` wins, then `--text-file`, then `CLICKER_TEXT`, then
    `CLICKER_TEXT_FILE`. None means the operator supplied nothing. A present empty
    string is empty text, not a missing one. Files are read as UTF-8 and kept
    byte for byte, including a trailing newline.
    """
    if text is not None and path is not None:
        raise SuppliedError("pass only one of --text and --text-file")
    if text is not None:
        return text
    if path is not None:
        return _read(path)
    if "CLICKER_TEXT" in os.environ:
        return os.environ["CLICKER_TEXT"]
    file_env = os.environ.get("CLICKER_TEXT_FILE", "").strip()
    if file_env:
        return _read(Path(file_env))
    return None


def supplied_url(url: str | None) -> str | None:
    """The exact address to open. `--open-url` wins over `CLICKER_OPEN_URL`."""
    if url is not None:
        return url
    env = os.environ.get("CLICKER_OPEN_URL")
    return env if env else None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise SuppliedError(f"text file not found: {path}") from e
    except UnicodeError as e:
        raise SuppliedError(f"text file is not UTF-8: {path}") from e
    except OSError as e:
        raise SuppliedError(f"could not read text file {path}: {e}") from e
