from __future__ import annotations

import os
import platform
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Best-effort global seeding for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Older torch versions may not support this
                pass
    except Exception:
        pass


@dataclass(frozen=True)
class EnvSnapshot:
    created_at_utc: str
    python_version: str
    platform: str
    pip_freeze_path: str


def write_pip_freeze(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        txt = subprocess.check_output(
            ["python", "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
        )
    except Exception as e:
        txt = f"# pip freeze failed: {type(e).__name__}: {e}\n"
    out_path.write_text(txt, encoding="utf-8")
    return out_path


def capture_env_snapshot(run_dir: Path) -> EnvSnapshot:
    """
    Write environment snapshot files under run_dir and return metadata.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    pip_path = write_pip_freeze(run_dir / "pip_freeze.txt")
    return EnvSnapshot(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        pip_freeze_path=str(pip_path),
    )

