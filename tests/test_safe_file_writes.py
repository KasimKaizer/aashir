import errno
import json
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import IO, Final, NoReturn, Self

import pytest

from aashir.tools import execute_tool_call

_ORIGINAL_TEXT: Final = "original text"
_REPLACED_BYTES: Final = b"replaced text"
# Group-writable, so a typical umask would strip a bit from a newly created file.
_ORIGINAL_MODE: Final = 0o664

_POSIX_ONLY: Final = pytest.mark.skipif(
    sys.platform == "win32", reason="needs POSIX links, FIFOs or directory flushing"
)

type _FaultInstaller = Callable[[pytest.MonkeyPatch], None]


# The fault installers below replace only the stdlib I/O calls that file
# replacement relies on, and only while the tool call under test runs.


class _DiskFullFile:
    """A temporary-file writer whose disk fills halfway through a write."""

    def __init__(self, real_file: IO[bytes]) -> None:
        self._real_file = real_file

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._real_file.close()

    def fileno(self) -> int:
        return self._real_file.fileno()

    def write(self, data: bytes) -> int:
        self._real_file.write(data[: len(data) // 2])
        raise OSError(errno.ENOSPC, "No space left on device")


def _fail_temporary_write(scoped: pytest.MonkeyPatch) -> None:
    real_fdopen = os.fdopen

    def fdopen(descriptor: int, mode: str) -> _DiskFullFile:
        return _DiskFullFile(real_fdopen(descriptor, "wb"))

    scoped.setattr(os, "fdopen", fdopen)


def _fail_temporary_flush(scoped: pytest.MonkeyPatch) -> None:
    def fsync(descriptor: int) -> NoReturn:
        raise OSError(errno.EIO, "Input/output error")

    scoped.setattr(os, "fsync", fsync)


def _fail_rename(scoped: pytest.MonkeyPatch) -> None:
    def replace(source: Path, destination: Path) -> NoReturn:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    scoped.setattr(os, "replace", replace)


def _cancel_before_rename(scoped: pytest.MonkeyPatch) -> None:
    def replace(source: Path, destination: Path) -> NoReturn:
        raise KeyboardInterrupt

    scoped.setattr(os, "replace", replace)


def _fail_directory_flush(scoped: pytest.MonkeyPatch, error: BaseException) -> None:
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise error
        real_fsync(descriptor)

    scoped.setattr(os, "fsync", fsync)


def _change_target_before_rename(scoped: pytest.MonkeyPatch, path: Path) -> None:
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        path.write_text("concurrent edit by someone else", encoding="utf-8")
        real_fsync(descriptor)

    scoped.setattr(os, "fsync", fsync)


def _change_request(tool_name: str, path: Path) -> str:
    """Build a request that turns ``original text`` into ``replaced text``."""
    if tool_name == "write_file":
        return json.dumps({"file_path": str(path), "content": "replaced text"})
    return json.dumps(
        {
            "file_path": str(path),
            "edits": [{"old_text": "original", "new_text": "replaced"}],
        }
    )


def _directory_entries(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


@pytest.fixture
def original_file(tmp_path: Path) -> Path:
    path = tmp_path / "note.txt"
    path.write_text(_ORIGINAL_TEXT, encoding="utf-8")
    path.chmod(_ORIGINAL_MODE)
    return path


_TOOL_NAMES: Final = [("write_file",), ("edit_file",)]
_PRE_REPLACEMENT_FAULTS: Final = [
    pytest.param(_fail_temporary_write, id="temporary-write"),
    pytest.param(_fail_temporary_flush, id="temporary-flush"),
    pytest.param(_fail_rename, id="rename"),
]


@pytest.mark.parametrize(("install_fault",), _PRE_REPLACEMENT_FAULTS)
@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_failure_before_replacement_keeps_original_bytes_and_mode(
    original_file: Path, tool_name: str, install_fault: _FaultInstaller
) -> None:
    # Given: an existing file with a non-default mode and a failing I/O step.
    mode_before = original_file.stat().st_mode
    request = _change_request(tool_name, original_file)

    # When: the tool tries to replace the file.
    with pytest.MonkeyPatch.context() as scoped:
        install_fault(scoped)
        result = execute_tool_call(tool_name, request)

    # Then: the failure is reported, the original is intact, and no temp file stays.
    assert result.startswith("error:")
    assert original_file.read_bytes() == _ORIGINAL_TEXT.encode()
    assert original_file.stat().st_mode == mode_before
    assert _directory_entries(original_file.parent) == ["note.txt"]


@pytest.mark.parametrize(("install_fault",), _PRE_REPLACEMENT_FAULTS)
def test_failed_write_of_a_new_file_leaves_no_destination_or_temp_file(
    tmp_path: Path, install_fault: _FaultInstaller
) -> None:
    # Given: a path that does not exist yet and a failing I/O step.
    path = tmp_path / "new.txt"
    request = json.dumps({"file_path": str(path), "content": "new content"})

    # When: write_file tries to create the file.
    with pytest.MonkeyPatch.context() as scoped:
        install_fault(scoped)
        result = execute_tool_call("write_file", request)

    # Then: nothing is left behind.
    assert result.startswith("error:")
    assert _directory_entries(tmp_path) == []


@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_cancellation_before_replacement_keeps_original_and_removes_temp_file(
    original_file: Path, tool_name: str
) -> None:
    # Given: an existing file and a cancellation just before the rename.
    request = _change_request(tool_name, original_file)

    # When: the tool is interrupted.
    with pytest.MonkeyPatch.context() as scoped:
        _cancel_before_rename(scoped)
        with pytest.raises(KeyboardInterrupt):
            execute_tool_call(tool_name, request)

    # Then: the original is intact and the temporary file is gone.
    assert original_file.read_bytes() == _ORIGINAL_TEXT.encode()
    assert _directory_entries(original_file.parent) == ["note.txt"]


@_POSIX_ONLY
@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_cancellation_after_replacement_reports_that_the_file_changed(
    original_file: Path, tool_name: str
) -> None:
    # Given: a cancellation that arrives while flushing the directory after the rename.
    request = _change_request(tool_name, original_file)

    # When: the tool is interrupted.
    with pytest.MonkeyPatch.context() as scoped:
        _fail_directory_flush(scoped, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt) as interruption:
            execute_tool_call(tool_name, request)

    # Then: the interruption says the file was already replaced, and it was not
    # rolled back.
    assert interruption.value.__notes__ == [
        f"{original_file} was already replaced with the new content"
    ]
    assert original_file.read_bytes() == _REPLACED_BYTES


@_POSIX_ONLY
@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_directory_flush_failure_reports_replacement_without_rolling_back(
    original_file: Path, tool_name: str
) -> None:
    # Given: a directory flush that fails after the rename.
    request = _change_request(tool_name, original_file)

    # When: the tool replaces the file.
    with pytest.MonkeyPatch.context() as scoped:
        _fail_directory_flush(scoped, OSError(errno.EIO, "Input/output error"))
        result = execute_tool_call(tool_name, request)

    # Then: the error states that the new content is in place but unconfirmed.
    assert result.startswith("error:")
    assert "was replaced with the new content" in result
    assert "may not survive a crash" in result
    assert original_file.read_bytes() == _REPLACED_BYTES


@pytest.mark.parametrize(
    ("tool_name", "original_text"),
    [
        pytest.param("write_file", _ORIGINAL_TEXT, id="write-existing"),
        pytest.param("edit_file", _ORIGINAL_TEXT, id="edit-existing"),
        pytest.param("write_file", None, id="write-new"),
    ],
)
def test_changed_target_is_refused_without_overwriting_the_other_change(
    tmp_path: Path, tool_name: str, original_text: str | None
) -> None:
    # Given: a target that another writer changes while the replacement is prepared.
    path = tmp_path / "note.txt"
    if original_text is not None:
        path.write_text(original_text, encoding="utf-8")
    request = _change_request(tool_name, path)

    # When: the tool tries to replace it.
    with pytest.MonkeyPatch.context() as scoped:
        _change_target_before_rename(scoped, path)
        result = execute_tool_call(tool_name, request)

    # Then: the other writer's content survives and no temp file stays.
    assert result.startswith("error:")
    assert "changed while the replacement was being prepared" in result
    assert path.read_text(encoding="utf-8") == "concurrent edit by someone else"
    assert _directory_entries(tmp_path) == ["note.txt"]


@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_successful_replacement_changes_only_the_content_and_keeps_the_mode(
    original_file: Path, tool_name: str
) -> None:
    # Given: an existing file with a non-default mode.
    mode_before = original_file.stat().st_mode
    request = _change_request(tool_name, original_file)

    # When: the tool replaces it.
    result = execute_tool_call(tool_name, request)

    # Then: the new content is in place, the mode is kept and no temp file stays.
    assert result.startswith("success:")
    assert original_file.read_bytes() == _REPLACED_BYTES
    assert original_file.stat().st_mode == mode_before
    assert _directory_entries(original_file.parent) == ["note.txt"]


def test_new_file_gets_the_normal_creation_mode(tmp_path: Path) -> None:
    # Given: the mode this process gives an ordinary new file.
    reference = tmp_path / "reference.txt"
    reference.write_text("", encoding="utf-8")

    # When: write_file creates a new file.
    execute_tool_call(
        "write_file",
        json.dumps({"file_path": str(tmp_path / "new.txt"), "content": "text"}),
    )

    # Then: it has that same mode, not the private mode of its temporary file.
    assert (tmp_path / "new.txt").stat().st_mode == reference.stat().st_mode


@_POSIX_ONLY
@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_symbolic_link_is_refused_and_its_target_is_unchanged(
    tmp_path: Path, tool_name: str
) -> None:
    # Given: a symbolic link to a file with existing content.
    target = tmp_path / "target.txt"
    target.write_text(_ORIGINAL_TEXT, encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    # When: the tool is asked to change the link.
    result = execute_tool_call(tool_name, _change_request(tool_name, link))

    # Then: it refuses, the target keeps its bytes, and the link stays a link.
    assert result.startswith("error:")
    assert "symbolic link" in result
    assert target.read_bytes() == _ORIGINAL_TEXT.encode()
    assert link.is_symlink()


@_POSIX_ONLY
@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_dangling_symbolic_link_is_refused_without_creating_its_target(
    tmp_path: Path, tool_name: str
) -> None:
    # Given: a symbolic link whose target does not exist.
    link = tmp_path / "link.txt"
    link.symlink_to(tmp_path / "missing.txt")

    # When: the tool is asked to change the link.
    result = execute_tool_call(tool_name, _change_request(tool_name, link))

    # Then: it refuses and the missing target is not created.
    assert "symbolic link" in result
    assert not (tmp_path / "missing.txt").exists()


@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_hard_linked_file_is_refused_and_both_names_are_unchanged(
    tmp_path: Path, tool_name: str
) -> None:
    # Given: one file reachable through two names.
    path = tmp_path / "note.txt"
    path.write_text(_ORIGINAL_TEXT, encoding="utf-8")
    other_name = tmp_path / "other.txt"
    os.link(path, other_name)

    # When: the tool is asked to change one name.
    result = execute_tool_call(tool_name, _change_request(tool_name, path))

    # Then: it refuses and both names still hold the original bytes.
    assert "more than one hard link" in result
    assert path.read_bytes() == _ORIGINAL_TEXT.encode()
    assert other_name.read_bytes() == _ORIGINAL_TEXT.encode()


@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
@pytest.mark.parametrize(
    ("make_special_file",),
    [
        pytest.param(Path.mkdir, id="directory"),
        pytest.param(os.mkfifo, marks=_POSIX_ONLY, id="fifo"),
    ],
)
def test_special_file_is_refused_without_opening_it(
    tmp_path: Path, tool_name: str, make_special_file: Callable[[Path], None]
) -> None:
    # Given: a path that is a directory or a FIFO (opening a FIFO would block).
    path = tmp_path / "special"
    make_special_file(path)
    mode_before = path.stat().st_mode

    # When: the tool is asked to change it.
    result = execute_tool_call(tool_name, _change_request(tool_name, path))

    # Then: it refuses and the path is unchanged.
    assert "not a regular file" in result
    assert path.stat().st_mode == mode_before
    assert _directory_entries(tmp_path) == ["special"]


@pytest.mark.parametrize(("tool_name",), _TOOL_NAMES)
def test_read_only_file_is_refused_and_unchanged(
    original_file: Path, tool_name: str
) -> None:
    # Given: a file its owner has made read-only.
    original_file.chmod(0o444)

    # When: the tool is asked to change it.
    result = execute_tool_call(tool_name, _change_request(tool_name, original_file))

    # Then: it refuses instead of replacing a file the owner protected.
    assert "not writable" in result
    assert original_file.read_bytes() == _ORIGINAL_TEXT.encode()
