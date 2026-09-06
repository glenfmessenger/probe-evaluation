"""Error types for the fail-loud loaders."""
from pathlib import Path
from typing import Iterable, Optional


class DataLoadError(RuntimeError):
    """Raised whenever real evaluation data cannot be loaded as expected.

    The pipeline must never continue on synthetic or authored substitutes when
    this is raised; callers should let it propagate.
    """


def schema_error(path, expected_keys: Iterable[str], found_keys: Iterable[str], context: str = "") -> DataLoadError:
    """Build a DataLoadError describing a key/column mismatch."""
    msg = (
        f"{context + ': ' if context else ''}schema mismatch in {Path(path)}\n"
        f"  expected keys: {sorted(expected_keys)}\n"
        f"  found keys:    {sorted(found_keys)}"
    )
    return DataLoadError(msg)


def missing_file_error(path, what: str, hint: Optional[str] = None) -> DataLoadError:
    msg = f"{what} not found: {Path(path)} (resolved to {Path(path).resolve()})"
    if hint:
        msg += f"\n  {hint}"
    return DataLoadError(msg)


def empty_error(path, what: str) -> DataLoadError:
    return DataLoadError(f"{what} loaded zero records from {Path(path)}; refusing to continue (no fallback data is permitted)")
