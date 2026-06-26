from pathlib import Path
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "src/data_processing/convert_panasonic_mat_to_csv.py"
STRICT_PIPELINE = PROJECT_ROOT / "src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    converter = load_module(CONVERTER, "convert_panasonic_mat_to_csv")
    pipeline = load_module(STRICT_PIPELINE, "strict_matched_temperature_pipeline")

    print("Step 1/2: converting original Panasonic .mat files to clean CSV files.")
    converter.convert_all()

    print("Step 2/2: running strict matched 25degC/10degC/0degC SOC experiments.")
    pipeline.main()


if __name__ == "__main__":
    main()
