"""
Emotion Recognition CNN
------------------------
Custom deep CNN for 7-class facial emotion recognition on 48×48 grayscale images.
Architecture: 4× (Conv→BN→ReLU→MaxPool→Dropout) blocks + Dense head.

Design decisions:
  - BatchNormalization after each Conv for training stability
  - GlobalAveragePooling instead of Flatten to reduce parameters and overfitting
  - Residual connections in deeper blocks for gradient flow
  - L2 regularization on Dense layers
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

NUM_CLASSES = 7
IMG_SIZE = 48


def _conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    pool_size: int = 2,
    dropout_rate: float = 0.25,
    l2: float = 1e-4,
    name_prefix: str = "block",
) -> tf.Tensor:
    """Standard Conv → BN → ReLU → MaxPool → Dropout block."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name_prefix}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    x = layers.MaxPooling2D(pool_size, name=f"{name_prefix}_pool")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
    return x


def _residual_conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    dropout_rate: float = 0.25,
    l2: float = 1e-4,
    name_prefix: str = "res_block",
) -> tf.Tensor:
    """Residual Conv block (no pooling) for the last feature extraction stage."""
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, padding="same", use_bias=False,
            name=f"{name_prefix}_shortcut_conv"
        )(x)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)

    x = layers.Conv2D(
        filters, kernel_size, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2), name=f"{name_prefix}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
    x = layers.Conv2D(
        filters, kernel_size, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(l2), name=f"{name_prefix}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
    return x


def build_emotion_cnn(
    input_shape: tuple = (IMG_SIZE, IMG_SIZE, 1),
    num_classes: int = NUM_CLASSES,
    dense_units: int = 512,
    dense_dropout: float = 0.5,
    l2: float = 1e-4,
) -> tf.keras.Model:
    """
    Build and return the emotion recognition CNN.

    Architecture summary:
        Input (48,48,1)
        → Block1: Conv32 → BN → ReLU → MaxPool(2) → Drop(0.25)  → (24,24,32)
        → Block2: Conv64 → BN → ReLU → MaxPool(2) → Drop(0.25)  → (12,12,64)
        → Block3: Conv128 → BN → ReLU → MaxPool(2) → Drop(0.25) → (6,6,128)
        → Block4: Conv256 → BN → ReLU → MaxPool(2) → Drop(0.25) → (3,3,256)
        → ResBlock: Conv256 (residual, no pool)                  → (3,3,256)
        → GlobalAvgPool                                          → (256,)
        → Dense(512) → BN → ReLU → Drop(0.5)
        → Dense(7, softmax)

    Args:
        input_shape: Image dimensions, default (48, 48, 1) for grayscale FER-2013
        num_classes: Number of emotion categories (7 for FER-2013)
        dense_units: Units in the penultimate dense layer
        dense_dropout: Dropout rate before final classifier
        l2: L2 regularization coefficient

    Returns:
        Compiled tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    x = _conv_block(inputs, 32,  dropout_rate=0.25, l2=l2, name_prefix="block1")
    x = _conv_block(x,      64,  dropout_rate=0.25, l2=l2, name_prefix="block2")
    x = _conv_block(x,      128, dropout_rate=0.25, l2=l2, name_prefix="block3")
    x = _conv_block(x,      256, dropout_rate=0.25, l2=l2, name_prefix="block4")
    x = _residual_conv_block(x, 256, dropout_rate=0.25, l2=l2, name_prefix="res_block5")

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(
        dense_units, use_bias=False,
        kernel_regularizer=regularizers.l2(l2),
        name="dense_head"
    )(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Activation("relu", name="dense_relu")(x)
    x = layers.Dropout(dense_dropout, name="dense_drop")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EmotionCNN")
    return model


def compile_emotion_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Compile with Adam + categorical crossentropy + accuracy metric."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )
    return model


def model_summary_info(model: tf.keras.Model) -> Dict[str, Any]:
    """Return a dict of model statistics."""
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    return {
        "name": model.name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }


if __name__ == "__main__":
    model = build_emotion_cnn()
    compile_emotion_model(model)
    model.summary()
    info = model_summary_info(model)
    print(f"\nTrainable parameters: {info['trainable_params']:,}")
