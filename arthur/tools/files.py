from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 200_000
MAX_LISTING = 500

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".log", ".py", ".js", ".ts", ".html", ".css", ".sql",
}


class WorkspaceError(ValueError):
    pass


def default_workspace() -> Path:
    configured = os.getenv("ARTHUR_WORKSPACE")
    if configured:
        return Path(configured)
    return Path.home() / ".arthur" / "workspace"


class Workspace:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_workspace()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()

    def resolve(self, relative: str) -> Path:
        if not relative or not relative.strip():
            raise WorkspaceError("A path is required")
        if "\x00" in relative:
            raise WorkspaceError("Path contains a null byte")

        candidate = (self.root / relative).resolve()

        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(
                f"{relative!r} is outside the workspace. "
                "File tools may only touch the workspace directory."
            )
        return candidate

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def is_text(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_SUFFIXES

    def read(self, relative: str) -> dict[str, Any]:
        path = self.resolve(relative)
        if not path.exists():
            raise WorkspaceError(f"{relative!r} does not exist")
        if path.is_dir():
            raise WorkspaceError(f"{relative!r} is a directory, not a file")
        if not self.is_text(path):
            raise WorkspaceError(
                f"{path.suffix or 'that file type'} is not readable as text"
            )

        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            raise WorkspaceError(
                f"{relative!r} is {size} bytes; the limit is {MAX_READ_BYTES}"
            )

        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "path": self.relative(path),
            "bytes": size,
            "lines": text.count("\n") + 1 if text else 0,
            "content": text,
        }

    def write(self, relative: str, content: str, append: bool = False) -> dict[str, Any]:
        path = self.resolve(relative)
        if path == self.root:
            raise WorkspaceError("Refusing to write over the workspace root")
        if not self.is_text(path):
            raise WorkspaceError(
                f"{path.suffix or 'that file type'} is not writable as text"
            )

        payload = content.encode("utf-8")
        if len(payload) > MAX_WRITE_BYTES:
            raise WorkspaceError(
                f"Content is {len(payload)} bytes; the limit is {MAX_WRITE_BYTES}"
            )

        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(content)

        return {
            "path": self.relative(path),
            "bytes": path.stat().st_size,
            "existed": existed,
            "appended": append,
        }

    def list(self, relative: str = ".") -> dict[str, Any]:
        path = self.resolve(relative)
        if not path.exists():
            raise WorkspaceError(f"{relative!r} does not exist")
        if not path.is_dir():
            raise WorkspaceError(f"{relative!r} is a file, not a directory")

        entries = []
        for child in sorted(path.iterdir())[:MAX_LISTING]:
            entries.append(
                {
                    "path": self.relative(child),
                    "kind": "directory" if child.is_dir() else "file",
                    "bytes": child.stat().st_size if child.is_file() else None,
                }
            )
        return {"path": self.relative(path), "entries": entries, "count": len(entries)}

    def delete(self, relative: str) -> dict[str, Any]:
        path = self.resolve(relative)
        if path == self.root:
            raise WorkspaceError("Refusing to delete the workspace root")
        if not path.exists():
            return {"path": relative, "deleted": False}
        if path.is_dir():
            raise WorkspaceError(
                f"{relative!r} is a directory; only files can be deleted"
            )

        path.unlink()
        return {"path": relative, "deleted": True}

    def search(self, needle: str, limit: int = 50) -> dict[str, Any]:
        matches = []
        for path in sorted(self.root.rglob("*")):
            if len(matches) >= limit:
                break
            if not path.is_file() or not self.is_text(path):
                continue
            if path.stat().st_size > MAX_READ_BYTES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if needle.lower() in line.lower():
                    matches.append(
                        {
                            "path": self.relative(path),
                            "line": number,
                            "text": line.strip()[:200],
                        }
                    )
                    break
        return {"query": needle, "matches": matches, "count": len(matches)}


class ReadArgs(BaseModel):
    path: str = Field(min_length=1, max_length=400)


class WriteArgs(BaseModel):
    path: str = Field(min_length=1, max_length=400)
    content: str = Field(max_length=MAX_WRITE_BYTES)
    append: bool = False


class ListArgs(BaseModel):
    path: str = Field(default=".", max_length=400)


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)


def register(registry, workspace: Workspace) -> None:
    from arthur.tools.registry import Risk

    @registry.tool(
        name="read_file",
        description="Read a text file from the workspace.",
        parameters=ReadArgs,
        risk=Risk.READ_ONLY,
    )
    def read_file(args: ReadArgs) -> dict[str, Any]:
        return workspace.read(args.path)

    @registry.tool(
        name="list_files",
        description="List the files and folders in a workspace directory.",
        parameters=ListArgs,
        risk=Risk.READ_ONLY,
    )
    def list_files(args: ListArgs) -> dict[str, Any]:
        return workspace.list(args.path)

    @registry.tool(
        name="search_files",
        description="Find workspace files containing a phrase.",
        parameters=SearchArgs,
        risk=Risk.READ_ONLY,
        timeout_seconds=20.0,
    )
    def search_files(args: SearchArgs) -> dict[str, Any]:
        return workspace.search(args.query)

    @registry.tool(
        name="write_file",
        description="Write or append to a text file in the workspace.",
        parameters=WriteArgs,
        risk=Risk.WRITES,
    )
    def write_file(args: WriteArgs) -> dict[str, Any]:
        return workspace.write(args.path, args.content, args.append)

    @registry.tool(
        name="delete_file",
        description="Permanently delete a file from the workspace.",
        parameters=ReadArgs,
        risk=Risk.IRREVERSIBLE,
    )
    def delete_file(args: ReadArgs) -> dict[str, Any]:
        return workspace.delete(args.path)
