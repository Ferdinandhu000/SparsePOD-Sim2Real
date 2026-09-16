#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on Python search path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from sparse_pod_sim2real.training.trainer import main

if __name__ == '__main__':
    main()
