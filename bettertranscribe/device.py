"""Cross-platform compute backend / device selection.

Decides between the faster-whisper backend (CPU everywhere, CUDA on NVIDIA) and
the mlx-whisper backend (Apple Silicon, optional). Also wires the pip-installed
NVIDIA cuBLAS/cuDNN DLLs onto the Windows search path before ctranslate2 loads.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """Resolved compute plan."""

    name: str          # "faster" | "mlx"
    device: str        # "cuda" | "cpu" | "mps"
    compute_type: str  # ctranslate2 compute type, e.g. "float16" / "int8"


def add_nvidia_dlls() -> None:
    """Put pip-installed cuBLAS/cuDNN DLLs on the search path (Windows GPU only).

    Must run before ctranslate2/faster-whisper import or the lazy cuBLAS load
    fails. No-op when the nvidia-* wheels are not installed.
    """
    if platform.system() != "Windows":
        return
    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return
    base = list(spec.submodule_search_locations)[0]
    extra: list[str] = []
    for binpath in glob.glob(os.path.join(base, "*", "bin")):
        if os.path.isdir(binpath):
            extra.append(binpath)
            try:
                os.add_dll_directory(binpath)
            except (OSError, AttributeError):
                pass
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])


def _cuda_available() -> bool:
    """True if ctranslate2 reports a usable CUDA device."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 - any import/runtime issue means "no CUDA"
        return False


def _mlx_available() -> bool:
    return importlib.util.find_spec("mlx_whisper") is not None


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def resolve_backend(device: str = "auto", prefer_mlx: bool = True) -> Backend:
    """Pick a backend + device.

    device: "auto" | "cuda" | "cpu" | "mps".
    - auto: CUDA if present; else mlx-whisper on Apple Silicon (if installed and
      prefer_mlx); else CPU.
    - explicit values force that path (cuda still falls back to cpu at load time
      if the device errors -- see asr.load_model).
    """
    add_nvidia_dlls()

    if device == "cpu":
        return Backend("faster", "cpu", "int8")
    if device == "cuda":
        return Backend("faster", "cuda", "float16")
    if device == "mps":
        # faster-whisper/ctranslate2 has no Metal backend; mlx is the real path.
        if _mlx_available():
            return Backend("mlx", "mps", "float16")
        return Backend("faster", "cpu", "int8")

    # auto
    if _cuda_available():
        return Backend("faster", "cuda", "float16")
    if prefer_mlx and _is_apple_silicon() and _mlx_available():
        return Backend("mlx", "mps", "float16")
    return Backend("faster", "cpu", "int8")
