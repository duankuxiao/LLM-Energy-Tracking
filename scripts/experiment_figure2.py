"""Figure 2 experiments: named grid input scenarios and country aggregation."""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.figure2_data import generate_figure2_data


def run(context, output_path):
    entry = context.config.experiments["figure2"]
    return generate_figure2_data(**context.figure_arguments(), output_path=output_path,
                                 scenario=entry["scenario"], country_display_top_n=entry["country_display_top_n"])


if __name__ == "__main__":
    from scripts.run_experiments import main
    main(single_name="figure2")
