"""Draw Figure 1 from exported experiment data; no model or raw-data reads."""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.plot_figures import main

if __name__ == "__main__":
    main(single_number=1)
