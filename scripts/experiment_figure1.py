"""Figure 1 experiments: demand paths and complete energy attribution."""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.figure1_data import generate_figure1_data


def run(context, output_path):
    entry = context.config.experiments["figure1"]
    return generate_figure1_data(**context.figure_arguments(), output_path=output_path,
                                 scenarios=entry["scenarios"], policy=entry["policy"],
                                 idle_allocation_rule=context.config.common["idle_allocation_rule"])


if __name__ == "__main__":
    from scripts.run_experiments import main
    main(single_name="figure1")
