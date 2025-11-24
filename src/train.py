# src/train.py
from pathlib import Path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

from src.visualization.plots import plot_history

def train_model(model, train_ds, val_ds, cfg: dict, run_paths: dict, logger=None):
    
    training_cfg = cfg.get("training", {})
    epochs = training_cfg.get("epochs", 20)
    patience = training_cfg.get("early_stopping_patience", 6)
    reduce_lr = training_cfg.get("reduce_lr", True)

    models_dir = Path(run_paths["models"])
    histories_dir = Path(run_paths["histories"])
    plots_dir = Path(run_paths["plots"])

    # callbacks
    ckpt_path = models_dir / "best_model.h5"
    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        CSVLogger(str(histories_dir / "history.csv"))
    ]

    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1))

    if logger:
        logger.info(f"Training for {epochs} epochs. Checkpoints -> {ckpt_path}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final models
    final_h5 = models_dir / "final_model.h5"
    saved_model_dir = models_dir / "saved_model"
    model.save(final_h5)
    model.save(saved_model_dir, save_format="tf")

    if logger:
        logger.info(f"Saved final model to {final_h5} and SavedModel to {saved_model_dir}")

    # Save plots
    plot_history(history, plots_dir)
    if logger:
        logger.info(f"Saved training plots to {plots_dir}")

    return history
