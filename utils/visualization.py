"""
Visualization Utilities
------------------------
Plotting, overlay rendering, and visualization tools for
emotion detection and image captioning evaluation.

Includes:
  - Training curve plotting (accuracy, loss)
  - Confusion matrix heatmaps
  - Emotion overlay rendering on frames
  - Attention weight visualization
  - Caption comparison display
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try importing matplotlib (optional for headless environments)
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available — plotting functions disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


EMOTION_COLORS_RGB = {
    "angry":    (255, 50,  50),
    "disgust":  (50,  180, 50),
    "fear":     (180, 50,  180),
    "happy":    (255, 220, 0),
    "neutral":  (180, 180, 180),
    "sad":      (50,  100, 255),
    "surprise": (255, 140, 0),
}

EMOTION_COLORS_BGR = {
    k: (b, g, r) for k, (r, g, b) in EMOTION_COLORS_RGB.items()
}


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 5),
) -> Optional["plt.Figure"]:
    """
    Plot accuracy and loss training curves from a Keras History dict.

    Args:
        history:     Dict with keys like 'accuracy', 'val_accuracy', 'loss', 'val_loss'
        output_path: If provided, save figure to this path
        title:       Plot title
        figsize:     Figure size

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Cannot plot: matplotlib not installed")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Accuracy subplot
    if "accuracy" in history:
        epochs = range(1, len(history["accuracy"]) + 1)
        axes[0].plot(epochs, history["accuracy"], label="Train", linewidth=2, color="#7c3aed")
        if "val_accuracy" in history:
            axes[0].plot(epochs, history["val_accuracy"], label="Val", linewidth=2, color="#06b6d4")
        axes[0].set_title("Accuracy", fontsize=13)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Loss subplot
    if "loss" in history:
        epochs = range(1, len(history["loss"]) + 1)
        axes[1].plot(epochs, history["loss"], label="Train", linewidth=2, color="#7c3aed")
        if "val_loss" in history:
            axes[1].plot(epochs, history["val_loss"], label="Val", linewidth=2, color="#06b6d4")
        axes[1].set_title("Loss", fontsize=13)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curves saved to {output_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (9, 8),
    normalize: bool = False,
) -> Optional["plt.Figure"]:
    """
    Plot a confusion matrix heatmap.

    Args:
        cm:           Confusion matrix array (num_classes x num_classes)
        class_names:  List of class label names
        output_path:  If provided, save figure to this path
        title:        Plot title
        figsize:      Figure size
        normalize:    If True, normalize rows to show percentages

    Returns:
        matplotlib Figure or None
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Cannot plot: matplotlib not installed")
        return None

    if normalize:
        cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {output_path}")

    return fig


def draw_emotion_overlay(
    frame_bgr: np.ndarray,
    results: List[Dict],
    show_bars: bool = True,
    bar_width: int = 80,
    font_scale: float = 0.7,
) -> np.ndarray:
    """
    Draw emotion detection results as overlays on a BGR frame.

    Args:
        frame_bgr:  OpenCV BGR image
        results:    List of dicts with keys: bbox, emotion, confidence, all_scores
        show_bars:  Whether to draw top-3 emotion confidence bars
        bar_width:  Max bar width in pixels
        font_scale: Font scale for labels

    Returns:
        Annotated BGR frame (copy)
    """
    annotated = frame_bgr.copy()

    for result in results:
        x, y, w, h = result["bbox"]
        emotion = result["emotion"]
        confidence = result["confidence"]
        color = EMOTION_COLORS_BGR.get(emotion, (255, 255, 255))

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label background
        label = f"{emotion} {confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(annotated, (x, y - lh - 10), (x + lw + 6, y), color, -1)
        cv2.putText(
            annotated, label,
            (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Top-3 confidence bars
        if show_bars and "all_scores" in result:
            sorted_emotions = sorted(
                result["all_scores"].items(), key=lambda e: e[1], reverse=True
            )[:3]
            bar_x = x + w + 10
            for i, (emo, score) in enumerate(sorted_emotions):
                bar_len = int(score * bar_width)
                bar_color = EMOTION_COLORS_BGR.get(emo, (200, 200, 200))
                cv2.rectangle(
                    annotated,
                    (bar_x, y + i * 20),
                    (bar_x + bar_len, y + i * 20 + 14),
                    bar_color, -1,
                )
                cv2.putText(
                    annotated, f"{emo[:4]} {score:.0%}",
                    (bar_x + bar_len + 4, y + i * 20 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA,
                )

    return annotated


def draw_caption_bar(
    frame_bgr: np.ndarray,
    caption: str,
    bar_height: int = 60,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Draw a semi-transparent caption bar at the bottom of a frame.

    Args:
        frame_bgr:  OpenCV BGR image
        caption:    Caption text to display
        bar_height: Height of the caption bar in pixels
        alpha:      Transparency of the background bar

    Returns:
        Frame with caption overlay (copy)
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Truncate long captions
    display_text = caption[:77] + "..." if len(caption) > 80 else caption
    cv2.putText(
        frame, f"Caption: {display_text}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def plot_emotion_distribution(
    scores: Dict[str, float],
    output_path: Optional[str] = None,
    title: str = "Emotion Distribution",
    figsize: Tuple[int, int] = (8, 5),
) -> Optional["plt.Figure"]:
    """
    Plot a horizontal bar chart of emotion scores.

    Args:
        scores:      Dict mapping emotion names to scores (0-1)
        output_path: Optional save path
        title:       Plot title
        figsize:     Figure size

    Returns:
        matplotlib Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    emotions = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [
        tuple(c / 255 for c in EMOTION_COLORS_RGB.get(e, (180, 180, 180)))
        for e in emotions
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(emotions, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def create_comparison_grid(
    images: List[np.ndarray],
    captions: List[str],
    output_path: Optional[str] = None,
    cols: int = 3,
    figsize_per_image: Tuple[float, float] = (4, 4),
) -> Optional["plt.Figure"]:
    """
    Create a grid of images with their generated captions.

    Args:
        images:              List of RGB numpy arrays
        captions:            List of caption strings
        output_path:         Optional save path
        cols:                Number of columns in the grid
        figsize_per_image:   Size per image cell

    Returns:
        matplotlib Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_per_image[0], rows * figsize_per_image[1])
    )
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n:
                axes[i, j].imshow(images[idx])
                axes[i, j].set_title(captions[idx], fontsize=9, wrap=True)
            axes[i, j].axis("off")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison grid saved to {output_path}")

    return fig
