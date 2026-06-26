from pathlib import Path
import importlib.util


PIPELINE = Path(__file__).resolve().parents[3] / "src/data/temperature_experiments/run_strict_matched_temperature_pipeline_25C_10C_0C.py"


def main():
    spec = importlib.util.spec_from_file_location("strict_matched_temperature_pipeline", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.pd.read_csv(module.MANIFEST_PATH)
    device = module.base.get_device()
    rows_dir = module.OUTPUT_ROOT / "selected_rows"
    frames_by_temp = {}
    for temp in [25, 10, 0]:
        _, _, train_frames, test_frames = module.load_temperature_frames(manifest, temp, rows_dir)
        frames_by_temp[temp] = {"train": train_frames, "test": test_frames}
    df = module.run_training_suite(
        25,
        frames_by_temp[25]["train"],
        {temp: frames_by_temp[temp]["test"] for temp in [25, 10, 0]},
        device,
        "transfer_from_25degC_only",
        module.OUTPUT_ROOT / "temperature_transfer/25degC_to_10degC_0degC_parallel_line",
    )
    full = module.add_averages(df)
    out_dir = module.OUTPUT_ROOT / "temperature_transfer/25degC_to_10degC_0degC_parallel_line"
    full.to_csv(out_dir / "parallel_line_full_summary.csv", index=False)
    avg = full[full["split"].eq("test_average")].copy()
    avg.to_csv(out_dir / "parallel_line_test_average.csv", index=False)
    (out_dir / "parallel_line_test_average.md").write_text(module.to_markdown(avg), encoding="utf-8")
    print(avg.to_string(index=False))


if __name__ == "__main__":
    main()
