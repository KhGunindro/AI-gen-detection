# main.py
import yaml
from pathlib import Path
import datetime

from src.utils.logger import setup_logger
from src.data.data_loader import create_datasets
from src.models.cnn_model import build_model
from src.train import train_model
from src.evaluation.metrics import evaluate_model

def load_config(path: str = "./src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_run_dirs(base_runs: str = "runs", run_id: str = None):
    """
    Create run directories:
    runs/
        logs/<run_id>
        models/<run_id>
        plots/<run_id>
        reports/<run_id>
        histories/<run_id>
    Returns: dict with keys: logs, models, plots, reports, histories, root, run_id
    """
    base = Path(base_runs)
    base.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dirs = {}
    for folder in ("logs", "models", "plots", "reports", "histories"):
        path = base / folder / run_id
        path.mkdir(parents=True, exist_ok=True)
        run_dirs[folder] = path

    run_dirs["run_id"] = run_id
    run_dirs["root"] = base
    return run_dirs

def main(config_path: str = "./src/config/config.yaml"):
    # Load configuration
    cfg = load_config(config_path)

    # Create run directories (guaranteed to include "plots")
    run_paths = make_run_dirs(base_runs=cfg.get("paths", {}).get("runs_dir", "runs"))

    # Ensure all keys exist
    for key in ["logs", "models", "plots", "reports", "histories"]:
        if key not in run_paths:
            run_paths[key] = Path("runs") / key / run_paths["run_id"]
            run_paths[key].mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(cfg, run_dir=run_paths["logs"])
    logger.info(f"Starting run {run_paths['run_id']}")

    # Create datasets
    train_ds, val_ds, test_ds, class_names = create_datasets(cfg, logger=logger)

    # Build model
    model = build_model(cfg, logger=logger)

    # Train model
    history = train_model(model, train_ds, val_ds, cfg, run_paths, logger=logger)

    # EVALUATION (FIXED ORDER)
    evaluate_model(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        run_dirs=run_paths,   
        cfg=cfg,             
        logger=logger
    )

    logger.info("Run finished.")

if __name__ == "__main__":
    main()
