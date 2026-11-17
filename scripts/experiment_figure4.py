"""Figure 4 experiments: configured sensitivity ranges and mechanism controls."""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.figure4_data import generate_figure4a_data


def run(context, output_path):
    entry = context.config.experiments["figure4"]
    return generate_figure4a_data(**context.figure_arguments(), output_path=output_path,
                                  scenario=entry["scenario"], policy=entry["policy"],
                                  hardware_config=context.config.hardware,
                                  idle_allocation_rule=context.config.common["idle_allocation_rule"],
                                  sensitivity_config=entry["sensitivity"])


if __name__ == "__main__":
    from scripts.run_experiments import main
    main(single_name="figure4")
