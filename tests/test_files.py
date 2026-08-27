from __future__ import annotations

import pytest

from arthur.tools.files import Workspace, WorkspaceError


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "workspace")


@pytest.mark.parametrize(
    "escape",
    [
        "../outside.txt",
        "../../outside.txt",
        "notes/../../outside.txt",
        "./../../outside.txt",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "\\\\server\\share\\file.txt",
    ],
)
def test_paths_outside_the_workspace_are_refused(workspace, escape):
    with pytest.raises(WorkspaceError):
        workspace.resolve(escape)


def test_the_dot_dot_dot_dot_bypass_stays_inside_the_workspace(workspace):
    resolved = workspace.resolve("....//....//outside.txt")

    assert workspace.root in resolved.parents


def test_a_symlink_pointing_outside_is_refused(workspace, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("classified", encoding="utf-8")
    link = workspace.root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available in this environment")

    with pytest.raises(WorkspaceError):
        workspace.read("link.txt")


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_path_is_refused(workspace, blank):
    with pytest.raises(WorkspaceError):
        workspace.resolve(blank)


def test_a_null_byte_in_the_path_is_refused(workspace):
    with pytest.raises(WorkspaceError):
        workspace.resolve("notes\x00.txt")


def test_paths_inside_the_workspace_are_allowed(workspace):
    resolved = workspace.resolve("notes/today.md")

    assert workspace.root in resolved.parents


def test_writing_then_reading_round_trips(workspace):
    workspace.write("notes.md", "# Hello\nworld\n")

    result = workspace.read("notes.md")

    assert result["content"] == "# Hello\nworld\n"
    assert result["path"] == "notes.md"
    assert result["bytes"] > 0


def test_writing_reports_whether_the_file_existed(workspace):
    assert workspace.write("a.txt", "one")["existed"] is False
    assert workspace.write("a.txt", "two")["existed"] is True


def test_writing_replaces_by_default(workspace):
    workspace.write("a.txt", "one")
    workspace.write("a.txt", "two")

    assert workspace.read("a.txt")["content"] == "two"


def test_appending_keeps_what_was_there(workspace):
    workspace.write("a.txt", "one")
    workspace.write("a.txt", "-two", append=True)

    assert workspace.read("a.txt")["content"] == "one-two"


def test_nested_directories_are_created_on_write(workspace):
    workspace.write("deep/nested/file.txt", "content")

    assert workspace.read("deep/nested/file.txt")["content"] == "content"


def test_reading_a_missing_file_is_a_clear_error(workspace):
    with pytest.raises(WorkspaceError, match="does not exist"):
        workspace.read("nope.txt")


def test_reading_a_directory_is_refused(workspace):
    workspace.write("dir/file.txt", "x")

    with pytest.raises(WorkspaceError, match="directory"):
        workspace.read("dir")


@pytest.mark.parametrize("suffix", [".exe", ".dll", ".bin", ".so", ".zip"])
def test_binary_file_types_are_refused(workspace, suffix):
    with pytest.raises(WorkspaceError, match="not readable as text|not writable as text"):
        workspace.write(f"payload{suffix}", "content")


def test_an_oversized_write_is_refused(workspace):
    with pytest.raises(WorkspaceError, match="limit is"):
        workspace.write("big.txt", "x" * 300_000)


def test_an_oversized_read_is_refused(workspace):
    path = workspace.root / "big.txt"
    path.write_text("x" * 300_000, encoding="utf-8")

    with pytest.raises(WorkspaceError, match="limit is"):
        workspace.read("big.txt")


def test_listing_shows_files_and_directories(workspace):
    workspace.write("a.txt", "one")
    workspace.write("sub/b.txt", "two")

    listing = workspace.list(".")
    kinds = {entry["path"]: entry["kind"] for entry in listing["entries"]}

    assert kinds["a.txt"] == "file"
    assert kinds["sub"] == "directory"


def test_listing_a_file_is_refused(workspace):
    workspace.write("a.txt", "one")

    with pytest.raises(WorkspaceError, match="not a directory"):
        workspace.list("a.txt")


def test_listing_a_missing_directory_is_refused(workspace):
    with pytest.raises(WorkspaceError, match="does not exist"):
        workspace.list("nowhere")


def test_deleting_removes_the_file(workspace):
    workspace.write("a.txt", "one")

    assert workspace.delete("a.txt")["deleted"] is True
    with pytest.raises(WorkspaceError):
        workspace.read("a.txt")


def test_deleting_something_absent_reports_false(workspace):
    assert workspace.delete("never-existed.txt")["deleted"] is False


def test_deleting_a_directory_is_refused(workspace):
    workspace.write("dir/file.txt", "x")

    with pytest.raises(WorkspaceError, match="only files"):
        workspace.delete("dir")


def test_the_workspace_root_cannot_be_deleted(workspace):
    with pytest.raises(WorkspaceError, match="workspace root"):
        workspace.delete(".")


def test_the_workspace_root_cannot_be_overwritten(workspace):
    with pytest.raises(WorkspaceError):
        workspace.write(".", "content")


def test_search_finds_a_phrase(workspace):
    workspace.write("a.md", "nothing here")
    workspace.write("b.md", "the answer is 42")

    result = workspace.search("answer")

    assert result["count"] == 1
    assert result["matches"][0]["path"] == "b.md"
    assert result["matches"][0]["line"] == 1


def test_search_is_case_insensitive(workspace):
    workspace.write("a.md", "The Answer")

    assert workspace.search("answer")["count"] == 1


def test_search_reports_nothing_when_there_is_nothing(workspace):
    workspace.write("a.md", "unrelated")

    assert workspace.search("missing")["count"] == 0


def test_search_reports_one_match_per_file(workspace):
    workspace.write("a.md", "answer\nanswer\nanswer")

    assert workspace.search("answer")["count"] == 1


def test_the_workspace_is_created_if_absent(tmp_path):
    root = tmp_path / "does" / "not" / "exist"
    space = Workspace(root)

    assert space.root.exists()
    assert space.root.is_dir()


def test_two_workspaces_do_not_see_each_other(tmp_path):
    first = Workspace(tmp_path / "one")
    second = Workspace(tmp_path / "two")
    first.write("a.txt", "mine")

    with pytest.raises(WorkspaceError):
        second.read("a.txt")
