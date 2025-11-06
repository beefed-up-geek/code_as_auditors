import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

LEGAL_CODE_PATH_EXAMPLE = "/Users/taeyoonkwack/Documents/code_as_auditors/method/outputs/legal_code_output/20251104_184338_5_5_5"
DEBUG = True
