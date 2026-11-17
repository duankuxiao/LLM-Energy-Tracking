"""Figure 3 experiments: endpoints, ordered bridge, activity and timing controls."""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.figure3_data import generate_figure3_data


def run(context, output_path):
    return generate_figure3_data(**context.figure_arguments(), output_path=output_path,
                                 activity_config=context.config.activity,
                                 reference_sensitivity_config=context.config.experiments["figure3"].get("reference_sensitivity"))


if __name__ == "__main__":
    from scripts.run_experiments import main
    main(single_name="figure3")
