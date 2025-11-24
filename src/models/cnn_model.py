import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers

def build_model(cfg: dict, logger=None):

    m = cfg["model"]
    input_shape = (
        m.get("input_height", 224),
        m.get("input_width", 224),
        m.get("input_channels", 3)
    )
    lr = m.get("learning_rate", 1e-4)
    loss = m.get("loss", "binary_crossentropy")
    metrics = m.get("metrics", ["accuracy"])

    if logger:
        logger.info(f"Building simplified model with input_shape={input_shape}")

    # NOTE: L2 Regularizer removed to prevent underfitting
    
    model = models.Sequential([
        keras.Input(shape=input_shape),

        # --- Block 1 ---
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        # Removed SpatialDropout2D

        # --- Block 2 ---
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        # Removed SpatialDropout2D

        # --- Block 3 ---
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        # Removed SpatialDropout2D

        # --- Head ---
        layers.GlobalAveragePooling2D(),
        
        # Reduced Dropout from 0.5 to 0.2 to allow learning
        layers.Dropout(0.2), 

        layers.Dense(64, activation="relu"),
        
        # Final safety dropout
        layers.Dropout(0.2), 

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )

    if logger:
        model.summary(print_fn=lambda s: logger.info(s))

    return model