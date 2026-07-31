"""Attest and isolate the executable runtime used for final figure publication."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


TRUSTED_EXECUTABLE_DIRECTORIES = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path("/usr/bin"),
)
POPPLER_TOOLS = ("pdfinfo", "pdftocairo", "pdfunite", "pdfimages", "pdffonts")
MATPLOTLIBRC = """\
backend: Agg
figure.facecolor: white
axes.facecolor: white
savefig.facecolor: white
savefig.transparent: False
font.family: sans-serif
font.sans-serif: DejaVu Sans
pdf.fonttype: 42
ps.fonttype: 42
svg.fonttype: none
text.usetex: False
timezone: UTC
"""


class TrustedFigureRuntimeError(RuntimeError):
    """Raised when the final-figure execution runtime cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ToolRecord:
    name: str
    path: str
    sha256: str
    version: str


@dataclass(frozen=True)
class TrustedFigureRuntime:
    tools: Mapping[str, ToolRecord]
    environment: Mapping[str, str]
    attestation: Mapping[str, object]

    def command(self, name: str, *arguments: str) -> tuple[str, ...]:
        try:
            tool = self.tools[name]
        except KeyError as exc:
            raise TrustedFigureRuntimeError(f"Unattested PDF tool: {name}") from exc
        return (tool.path, *arguments)

    def run(
        self,
        name: str,
        arguments: Sequence[str],
        *,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        record = self.tools[name]
        path = _validate_executable(Path(record.path), name)
        if str(path) != record.path or _sha256(path) != record.sha256:
            raise TrustedFigureRuntimeError(
                f"Attested executable changed after runtime creation: {name}"
            )
        return subprocess.run(
            self.command(name, *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(self.environment),
        )


def _validate_executable(path: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise TrustedFigureRuntimeError(
            f"Could not resolve trusted executable {name}: {path}"
        ) from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise TrustedFigureRuntimeError(
            f"Trusted executable is not a regular executable: {resolved}"
        )
    if info.st_uid not in {0, os.getuid()}:
        raise TrustedFigureRuntimeError(
            f"Trusted executable has an unexpected owner: {resolved}"
        )
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise TrustedFigureRuntimeError(
            f"Trusted executable is group/world writable: {resolved}"
        )
    return resolved


def _resolve_executable(name: str) -> Path:
    for directory in TRUSTED_EXECUTABLE_DIRECTORIES:
        candidate = directory / name
        if candidate.exists() or candidate.is_symlink():
            return _validate_executable(candidate, name)
    raise TrustedFigureRuntimeError(
        f"Required Poppler tool was not found in trusted directories: {name}"
    )


def _base_environment(runtime_root: Path) -> dict[str, str]:
    home = runtime_root / "home"
    xdg_config = runtime_root / "xdg" / "config"
    xdg_cache = runtime_root / "xdg" / "cache"
    mpl_config = runtime_root / "matplotlib"
    texmf_home = runtime_root / "texmf" / "home"
    texmf_config = runtime_root / "texmf" / "config"
    texmf_var = runtime_root / "texmf" / "var"
    texmf_cache = runtime_root / "texmf" / "cache"
    texmf_output = runtime_root / "texmf" / "output"
    temporary = runtime_root / "tmp"
    for directory in (
        home,
        xdg_config,
        xdg_cache,
        mpl_config,
        texmf_home,
        texmf_config,
        texmf_var,
        texmf_cache,
        texmf_output,
        temporary,
    ):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    return {
        "PATH": os.defpath,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(xdg_config),
        "XDG_CACHE_HOME": str(xdg_cache),
        "MPLCONFIGDIR": str(mpl_config),
        "MPLBACKEND": "Agg",
        "FONTCONFIG_FILE": str(runtime_root / "fontconfig" / "fonts.conf"),
        "FONTCONFIG_PATH": str(runtime_root / "fontconfig"),
        "TEXMFHOME": str(texmf_home),
        "TEXMFCONFIG": str(texmf_config),
        "TEXMFVAR": str(texmf_var),
        "TEXMFCACHE": str(texmf_cache),
        "TEXMFOUTPUT": str(texmf_output),
        "TEXINPUTS": "",
        "BIBINPUTS": "",
        "BSTINPUTS": "",
        "TMPDIR": str(temporary),
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "PYTHONUTF8": "1",
        "SOURCE_DATE_EPOCH": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def _package_record(name: str) -> dict[str, str]:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", None)
    if name == "reportlab":
        version = getattr(module, "Version", version)
    return {
        "name": name,
        "version": str(version or "unknown"),
        "module_path": str(Path(module.__file__).resolve()),
    }


def _copy_fonts(runtime_root: Path) -> list[dict[str, str]]:
    import matplotlib
    import reportlab
    from fontTools.ttLib import TTFont

    sources = [
        Path(matplotlib.get_data_path()).resolve() / "fonts" / "ttf" / "DejaVuSans.ttf",
        Path(matplotlib.get_data_path()).resolve()
        / "fonts"
        / "ttf"
        / "DejaVuSans-Bold.ttf",
        Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf",
        Path(reportlab.__file__).resolve().parent / "fonts" / "VeraBd.ttf",
    ]
    font_root = runtime_root / "fonts"
    font_root.mkdir(mode=0o700)
    records = []
    for source in sources:
        if not source.is_file() or source.is_symlink():
            raise TrustedFigureRuntimeError(f"Required font is unavailable: {source}")
        target = font_root / source.name
        target.write_bytes(source.read_bytes())
        target.chmod(0o400)
        font = TTFont(target, lazy=True)
        names = {}
        for record in font["name"].names:
            if record.nameID in {1, 2, 5} and record.nameID not in names:
                try:
                    names[record.nameID] = record.toUnicode()
                except UnicodeError:
                    continue
        font.close()
        records.append(
            {
                "name": source.name,
                "family": names.get(1, "unknown"),
                "style": names.get(2, "unknown"),
                "version": names.get(5, "unknown"),
                "source_path": str(source),
                "sha256": _sha256(target),
            }
        )
    return records


def _write_configuration(runtime_root: Path) -> dict[str, str]:
    matplotlibrc = runtime_root / "matplotlib" / "matplotlibrc"
    matplotlibrc.write_text(MATPLOTLIBRC, encoding="ascii")
    matplotlibrc.chmod(0o400)
    fontconfig = runtime_root / "fontconfig"
    fontconfig.mkdir(mode=0o700)
    fonts_conf = fontconfig / "fonts.conf"
    fonts_conf.write_text(
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
        "<fontconfig>\n"
        f"  <dir>{runtime_root / 'fonts'}</dir>\n"
        f"  <cachedir>{runtime_root / 'fontconfig-cache'}</cachedir>\n"
        "</fontconfig>\n",
        encoding="utf-8",
    )
    fonts_conf.chmod(0o400)
    return {
        "matplotlibrc_sha256": _sha256(matplotlibrc),
        "fontconfig_sha256": _sha256(fonts_conf),
        "matplotlibrc": MATPLOTLIBRC,
        "fontconfig_policy": "private copied fonts only",
    }


def _tool_version(path: Path, environment: Mapping[str, str]) -> str:
    try:
        completed = subprocess.run(
            [str(path), "-v"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=dict(environment),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrustedFigureRuntimeError(
            f"Could not query trusted tool version: {path}"
        ) from exc
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    first_line = next(
        (line.strip() for line in output.splitlines() if line.strip()), ""
    )
    if not first_line or not re.search(r"\d", first_line):
        raise TrustedFigureRuntimeError(
            f"Trusted tool did not report a version: {path}"
        )
    return first_line


def prepare_trusted_figure_runtime(runtime_root: Path) -> TrustedFigureRuntime:
    """Create one private deterministic runtime and attest every external tool."""

    runtime_root = Path(runtime_root).resolve()
    if runtime_root.exists():
        raise TrustedFigureRuntimeError(
            f"Trusted runtime root must not exist: {runtime_root}"
        )
    runtime_root.mkdir(parents=True, mode=0o700)
    runtime_root.chmod(0o700)
    environment = _base_environment(runtime_root)
    fonts = _copy_fonts(runtime_root)
    configuration = _write_configuration(runtime_root)
    python_executable = _validate_executable(Path(sys.executable), "python")
    tools = {}
    for name in POPPLER_TOOLS:
        path = _resolve_executable(name)
        record = ToolRecord(
            name=name,
            path=str(path),
            sha256=_sha256(path),
            version=_tool_version(path, environment),
        )
        tools[name] = record
    attestation = {
        "schema_version": 1,
        "python": {
            "executable": str(python_executable),
            "sha256": _sha256(python_executable),
            "version": sys.version.splitlines()[0],
        },
        "packages": [
            _package_record(name) for name in ("matplotlib", "PIL", "reportlab")
        ],
        "tools": [asdict(tools[name]) for name in POPPLER_TOOLS],
        "fonts": fonts,
        "configuration": configuration,
        "deterministic_environment": {
            key: value
            for key, value in environment.items()
            if key
            in {
                "LC_ALL",
                "LANG",
                "TZ",
                "PYTHONHASHSEED",
                "PYTHONUTF8",
                "SOURCE_DATE_EPOCH",
                "MPLBACKEND",
            }
        },
    }
    return TrustedFigureRuntime(
        tools=tools,
        environment=environment,
        attestation=attestation,
    )


def run_attested_tool(
    runtime: TrustedFigureRuntime,
    name: str,
    arguments: Sequence[str],
    *,
    timeout: int = 60,
) -> str:
    try:
        completed = runtime.run(name, arguments, timeout=timeout)
    except FileNotFoundError as exc:
        raise TrustedFigureRuntimeError(
            f"Attested executable disappeared: {runtime.tools[name].path}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown error").strip()
        raise TrustedFigureRuntimeError(
            f"{name} failed under the trusted runtime: {detail}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TrustedFigureRuntimeError(f"{name} timed out") from exc
    return completed.stdout
