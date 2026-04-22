from deepagents.backends import LocalShellBackend
from pathlib import Path
import os

sandbox_dir = Path("env/").absolute()
backend = LocalShellBackend(
    root_dir=sandbox_dir,
    virtual_mode=True,
    env={
        "PATH": os.environ["PATH"]
    },
    # inherit_env=True,
    timeout=120,
)