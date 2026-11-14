"""Replacement of host files that keeps the original when a write fails."""

import os
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

_TEMPORARY_FILE_PREFIX: Final = ".aashir-"
_NEW_FILE_MODE: Final = 0o666


class _FileReplacementError(Exception):
    """Base for replacement failures that explain themselves to the model."""


@dataclass(frozen=True, slots=True)
class _ReplacementRefusedError(_FileReplacementError):
    """The target was left unchanged because replacing it is unsafe."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        return f"refusing to replace {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class _ReplacedWithoutFlushError(_FileReplacementError):
    """The file was replaced, but the rename may not survive a crash."""

    path: Path
    cause: OSError

    @override
    def __str__(self) -> str:
        return (
            f"{self.path} was replaced with the new content, but flushing its "
            f"directory failed ({self.cause}); the new content may not survive a crash"
        )


@dataclass(frozen=True, slots=True)
class _FileState:
    """Identity and metadata used to detect a concurrent change to a file."""

    device: int
    inode: int
    size: int
    modified_ns: int
    mode: int
    owner: int
    group: int


def _observe_file(path: Path) -> _FileState | None:
    """Describe an existing file that can be replaced.

    A symbolic link, special file, multiply linked file or read-only file is
    refused rather than replaced with a file of a different meaning.

    Args:
        path: File to inspect. Only its final component is checked.

    Returns:
        The file's state, or ``None`` if it does not exist.

    Raises:
        _ReplacementRefusedError: If the path is not a file that can be replaced.
        OSError: If the file cannot be inspected.
    """
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None

    if stat.S_ISLNK(status.st_mode):
        raise _ReplacementRefusedError(path, "it is a symbolic link")
    if not stat.S_ISREG(status.st_mode):
        raise _ReplacementRefusedError(path, "it is not a regular file")
    if status.st_nlink > 1:
        raise _ReplacementRefusedError(path, "it has more than one hard link")
    if not os.access(path, os.W_OK):
        raise _ReplacementRefusedError(path, "it is not writable")

    return _FileState(
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        modified_ns=status.st_mtime_ns,
        mode=status.st_mode,
        owner=status.st_uid,
        group=status.st_gid,
    )


def _replace_file(path: Path, content: str, expected: _FileState | None) -> None:
    """Replace a file with UTF-8 text, keeping the original if anything fails.

    The content is written and fsynced to a temporary file in the same
    directory, then renamed over the target. A failure or cancellation before
    the rename leaves the original untouched and deletes the temporary file.
    An existing file keeps its permission bits; a new file gets the default
    mode. ACLs and extended attributes are not preserved.

    The target is re-checked just before the rename, but another process can
    still modify it between that check and the rename, and that change is lost.

    Windows behaves differently in these ways:
    - Only the read-only attribute is preserved and checked. There are no Unix
      permission bits.
    - The owner and group check never refuses, since ``st_uid`` and ``st_gid``
      are always 0.
    - The directory is not fsynced, because Windows cannot open a directory.
      After a crash, the replacement may be lost.
    - If another process has the target open, the rename raises
      ``PermissionError`` and the original is kept.

    Args:
        path: Target file.
        content: Complete new text. Newline sequences are written unchanged.
        expected: State from ``_observe_file`` before the content was prepared.
            ``None`` means the file must still be absent.

    Raises:
        _ReplacementRefusedError: If the target changed, or its owner or group
            would change.
        _ReplacedWithoutFlushError: If the rename succeeded but fsyncing the
            directory failed.
        KeyboardInterrupt: If cancelled while fsyncing the directory, after the
            rename. A note on the exception says the file was replaced.
        OSError: If the temporary file cannot be written or renamed.
    """
    data = content.encode("utf-8")
    temporary_path = path.with_name(
        f"{_TEMPORARY_FILE_PREFIX}{secrets.token_hex(8)}.tmp"
    )
    # Create with the original mode so the temporary file is never more
    # accessible than the original, even briefly.
    creation_mode = _NEW_FILE_MODE if expected is None else stat.S_IMODE(expected.mode)
    descriptor = os.open(
        temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, creation_mode
    )

    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            if expected is not None:
                status = os.fstat(temporary_file.fileno())
                if (status.st_uid, status.st_gid) != (expected.owner, expected.group):
                    raise _ReplacementRefusedError(
                        path, "its owner or group would change"
                    )
                os.fchmod(temporary_file.fileno(), stat.S_IMODE(expected.mode))
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        if _observe_file(path) != expected:
            raise _ReplacementRefusedError(
                path, "it changed while the replacement was being prepared"
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

    _flush_directory(path)


def _flush_directory(path: Path) -> None:
    """Fsync the directory of ``path`` so the rename survives a crash."""
    if sys.platform == "win32":
        # Windows cannot open a directory for flushing with this approach.
        return

    try:
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise _ReplacedWithoutFlushError(path, error) from error
    except KeyboardInterrupt as interrupt:
        interrupt.add_note(f"{path} was already replaced with the new content")
        raise
