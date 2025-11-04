import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from method.utils.law_parser import full_law_json
from method.utils.llm_interface import llm_response

INPUT_PSEUDOCODE_DIR = "./pseudocode_output/20251104_053331_gpt5"
DEBUG = True

def main():
    return

if __name__ == "__main__":
    main()
