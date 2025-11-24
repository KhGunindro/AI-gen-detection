import tensorflow as tf
from pathlib import Path

def create_datasets(cfg: dict, logger=None):
    ds = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})

    # ---------------------------------------------------------
    # PATH SETUP: Assumes you have physically split the data
    # ---------------------------------------------------------
    # If your config just points to a main folder, we assume standard subfolders
    # Adjust these strings if your folders are named differently
    base_dir = Path(ds.get("train_dir", "dataset")).parent 
    
    train_dir = Path("dataset/train") 
    val_dir =   Path("dataset/val")
    test_dir =  Path("dataset/test")
    
    # Fallback: If user put exact paths in config, use those
    if Path(ds["train_dir"]).name == "train": train_dir = Path(ds["train_dir"])
    if ds.get("test_dir"): test_dir = Path(ds["test_dir"])
    if ds.get("val_dir"): val_dir = Path(ds["val_dir"])

    img_size = (ds["img_height"], ds["img_width"])
    batch_size = ds["batch_size"]
    
    if logger:
        logger.info(f"Train Dir: {train_dir}")
        logger.info(f"Val Dir:   {val_dir}")
        logger.info(f"Test Dir:  {test_dir}")

    # ---------------------------------------------------------
    # 1. LOAD TRAINING SET
    # ---------------------------------------------------------
    # shuffle=True is crucial here for stochastic gradient descent
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(train_dir),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=ds.get("seed", 42),
        label_mode="int"
    )

    # ---------------------------------------------------------
    # 2. LOAD VALIDATION SET (Physical Folder)
    # ---------------------------------------------------------
    # shuffle=False prevents evaluation jitter. 
    # No validation_split argument needed because it's a separate folder.
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(val_dir),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False, 
        label_mode="int"
    )

    # ---------------------------------------------------------
    # 3. LOAD TEST SET
    # ---------------------------------------------------------
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(test_dir),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="int"
    )

    class_names = train_ds.class_names
    if logger:
        logger.info(f"Classes detected: {class_names}")

    # ---------------------------------------------------------
    # 4. AUGMENTATION PIPELINE
    # ---------------------------------------------------------
    aug_layers = []
    if aug_cfg.get("flip_horizontal"):
        aug_layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if aug_cfg.get("rotation", 0) > 0:
        aug_layers.append(tf.keras.layers.RandomRotation(aug_cfg["rotation"]))
    if aug_cfg.get("zoom", 0) > 0:
        aug_layers.append(tf.keras.layers.RandomZoom(aug_cfg["zoom"]))

    data_augmentation = tf.keras.Sequential(aug_layers) if aug_layers else None
    
    # ---------------------------------------------------------
    # 5. PREPROCESSING (Resize, Augment, Normalize)
    # ---------------------------------------------------------
    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    AUTOTUNE = tf.data.AUTOTUNE

    def prepare(dataset, training=False):
        # Inner function to apply transformations per image
        def _apply(x, y):
            # 1. Resize
            x = tf.image.resize(x, img_size)
            
            # 2. Augment (Only if training)
            if training and data_augmentation:
                x = data_augmentation(x)
            
            # 3. Normalize (0 to 1)
            x = normalize(x)
            return x, y

        # Apply the mapping
        dataset = dataset.map(_apply, num_parallel_calls=AUTOTUNE)

        # Optimization steps
        if training:
            dataset = dataset.shuffle(1000)
            
        return dataset.prefetch(AUTOTUNE)

    # Return processed datasets
    return prepare(train_ds, True), prepare(val_ds, False), prepare(test_ds, False), class_names