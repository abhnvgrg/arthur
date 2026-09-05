from __future__ import annotations

import os
import platform
import shutil
import subprocess
import webbrowser
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MAX_CLIPBOARD = 4000
LAUNCH_TIMEOUT = 15.0

APP_ALIASES = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "explorer": "explorer.exe",
    "files": "explorer.exe",
    "file explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "terminal": "wt.exe",
    "cmd": "cmd.exe",
    "paint": "mspaint.exe",
    "camera": "microsoft.windows.camera:",
    "spotify": "spotify.exe",
    "chrome": "chrome.exe",
    "edge": "msedge.exe",
    "firefox": "firefox.exe",
    "code": "code.cmd",
    "vscode": "code.cmd",
}

MEDIA_KEYS = {
    "play": 0xB3,
    "pause": 0xB3,
    "playpause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
    "mute": 0xAD,
    "volume_up": 0xAF,
    "volume_down": 0xAE,
}

KEYEVENTF_KEYUP = 0x0002


class SystemError_(Exception):
    pass


def on_windows() -> bool:
    return platform.system() == "Windows"


def _require_windows(what: str) -> None:
    if not on_windows():
        raise SystemError_(f"{what} is only wired up for Windows on this machine")


def resolve_app(name: str) -> str:
    key = " ".join(name.strip().lower().split())
    if key in APP_ALIASES:
        return APP_ALIASES[key]
    if not key:
        raise SystemError_("Name an application to open")
    return key if key.endswith((".exe", ".cmd", ".bat", ":")) else f"{key}.exe"


def launch(target: str) -> dict[str, Any]:
    _require_windows("Launching applications")

    if target.endswith(":"):
        subprocess.Popen(["cmd", "/c", "start", "", target], shell=False)
        return {"launched": target, "how": "shell"}

    found = shutil.which(target)
    if found is None:
        subprocess.Popen(["cmd", "/c", "start", "", target], shell=False)
        return {"launched": target, "how": "shell"}

    subprocess.Popen([found])
    return {"launched": target, "path": found, "how": "direct"}


def press_media_key(code: int) -> None:
    _require_windows("Media keys")
    import ctypes

    ctypes.windll.user32.keybd_event(code, 0, 0, 0)
    ctypes.windll.user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)


def read_clipboard() -> str:
    try:
        import tkinter
    except ImportError as error:
        raise SystemError_("Clipboard access needs tkinter") from error

    root = tkinter.Tk()
    root.withdraw()
    try:
        return root.clipboard_get()[:MAX_CLIPBOARD]
    except tkinter.TclError:
        return ""
    finally:
        root.destroy()


def write_clipboard(text: str) -> None:
    try:
        import tkinter
    except ImportError as error:
        raise SystemError_("Clipboard access needs tkinter") from error

    root = tkinter.Tk()
    root.withdraw()
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
    finally:
        root.destroy()


def system_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processors": os.cpu_count(),
    }

    usage = shutil.disk_usage(Path.home().anchor or "/")
    stats["disk_total_gb"] = round(usage.total / 1_000_000_000, 1)
    stats["disk_free_gb"] = round(usage.free / 1_000_000_000, 1)

    try:
        import psutil
    except ImportError:
        stats["note"] = "Install psutil for cpu, memory and battery figures"
        return stats

    stats["cpu_percent"] = psutil.cpu_percent(interval=0.2)
    memory = psutil.virtual_memory()
    stats["memory_percent"] = memory.percent
    stats["memory_free_gb"] = round(memory.available / 1_000_000_000, 1)

    battery = getattr(psutil, "sensors_battery", lambda: None)()
    if battery is not None:
        stats["battery_percent"] = round(battery.percent)
        stats["charging"] = battery.power_plugged

    return stats


def take_screenshot(directory: Path | None = None) -> dict[str, Any]:
    try:
        from PIL import ImageGrab
    except ImportError as error:
        raise SystemError_(
            "Screenshots need pillow. pip install pillow"
        ) from error

    from datetime import datetime

    folder = directory or Path.home() / ".arthur" / "screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"screen_{datetime.now():%Y%m%d_%H%M%S}.png"

    image = ImageGrab.grab()
    image.save(path)
    return {"path": str(path), "width": image.width, "height": image.height}


def focus_window(title: str) -> dict[str, Any]:
    _require_windows("Focusing windows")
    import ctypes

    user32 = ctypes.windll.user32
    needle = title.strip().lower()
    found: list[tuple[int, str]] = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
    )

    def visit(handle, _):
        length = user32.GetWindowTextLengthW(handle)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(handle, buffer, length + 1)
            if needle in buffer.value.lower() and user32.IsWindowVisible(handle):
                found.append((handle, buffer.value))
        return True

    user32.EnumWindows(WNDENUMPROC(visit), None)

    if not found:
        return {"title": title, "focused": False, "reason": "no matching window"}

    handle, name = found[0]
    user32.ShowWindow(handle, 9)
    user32.SetForegroundWindow(handle)
    return {"title": name, "focused": True, "matches": len(found)}


class OpenAppArgs(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=80,
        description="Application name, for example 'notepad', 'spotify', 'chrome'",
    )


class FocusWindowArgs(BaseModel):
    title: str = Field(min_length=2, max_length=120, description="Part of the window title")


class MediaArgs(BaseModel):
    action: str = Field(
        description="One of: play, pause, next, previous, stop, mute, volume_up, volume_down"
    )


class ClipboardWriteArgs(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CLIPBOARD)


class OpenUrlArgs(BaseModel):
    url: str = Field(min_length=4, max_length=2000, description="An http or https URL")


class NoArgs(BaseModel):
    pass


def register(registry, screenshot_dir: Path | None = None) -> None:
    from arthur.tools.registry import Risk

    @registry.tool(
        name="system_stats",
        description="Report this machine's cpu, memory, disk and battery.",
        parameters=NoArgs,
        risk=Risk.READ_ONLY,
    )
    def stats(_: NoArgs) -> dict[str, Any]:
        return system_stats()

    @registry.tool(
        name="read_clipboard",
        description="Read whatever is currently on the clipboard.",
        parameters=NoArgs,
        risk=Risk.READ_ONLY,
    )
    def clipboard_in(_: NoArgs) -> dict[str, Any]:
        text = read_clipboard()
        return {"text": text, "empty": not text}

    @registry.tool(
        name="open_app",
        description="Open an application on this machine.",
        parameters=OpenAppArgs,
        risk=Risk.WRITES,
        timeout_seconds=LAUNCH_TIMEOUT,
    )
    def open_app(args: OpenAppArgs) -> dict[str, Any]:
        return launch(resolve_app(args.name))

    @registry.tool(
        name="open_url",
        description="Open a web page in the default browser.",
        parameters=OpenUrlArgs,
        risk=Risk.WRITES,
        timeout_seconds=LAUNCH_TIMEOUT,
    )
    def open_url(args: OpenUrlArgs) -> dict[str, Any]:
        if not args.url.lower().startswith(("http://", "https://")):
            raise SystemError_("Only http and https URLs can be opened")
        webbrowser.open(args.url)
        return {"url": args.url, "opened": True}

    @registry.tool(
        name="focus_window",
        description="Bring a window to the front by part of its title.",
        parameters=FocusWindowArgs,
        risk=Risk.WRITES,
    )
    def focus(args: FocusWindowArgs) -> dict[str, Any]:
        return focus_window(args.title)

    @registry.tool(
        name="media_control",
        description="Control media playback and volume with the media keys.",
        parameters=MediaArgs,
        risk=Risk.WRITES,
    )
    def media(args: MediaArgs) -> dict[str, Any]:
        action = args.action.strip().lower().replace(" ", "_")
        if action not in MEDIA_KEYS:
            raise SystemError_(
                f"Unknown media action {args.action!r}. Use one of: "
                f"{', '.join(sorted(MEDIA_KEYS))}"
            )
        press_media_key(MEDIA_KEYS[action])
        return {"action": action, "sent": True}

    @registry.tool(
        name="write_clipboard",
        description="Replace the clipboard contents with the given text.",
        parameters=ClipboardWriteArgs,
        risk=Risk.WRITES,
    )
    def clipboard_out(args: ClipboardWriteArgs) -> dict[str, Any]:
        write_clipboard(args.text)
        return {"length": len(args.text), "written": True}

    @registry.tool(
        name="take_screenshot",
        description="Capture the screen to a PNG file and return its path.",
        parameters=NoArgs,
        risk=Risk.WRITES,
        timeout_seconds=LAUNCH_TIMEOUT,
    )
    def screenshot(_: NoArgs) -> dict[str, Any]:
        return take_screenshot(screenshot_dir)
