"""
Evaluation Metrics
-------------------
BLEU (1-4), corpus-level and sentence-level.
Compatible with NLTK's bleu_score module.

Reference: Papineni et al. "BLEU: a Method for Automatic Evaluation of Machine Translation"
"""

import logging
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def compute_corpus_bleu(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
) -> Dict[str, float]:
    """
    Compute BLEU-1 through BLEU-4 at corpus level.

    Args:
        references:  List of reference lists per hypothesis.
                     Each element is a list of reference token lists.
                     Shape: [N, num_refs, ref_length]
        hypotheses:  List of hypothesis token lists.
                     Shape: [N, hyp_length]

    Returns:
        Dict with keys: bleu1, bleu2, bleu3, bleu4
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        raise ImportError("nltk not installed. Run: pip install nltk")

    smoother = SmoothingFunction().method1

    scores = {}
    weight_sets = {
        "bleu1": (1.0, 0.0, 0.0, 0.0),
        "bleu2": (0.5, 0.5, 0.0, 0.0),
        "bleu3": (1/3, 1/3, 1/3, 0.0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }

    for name, weights in weight_sets.items():
        score = corpus_bleu(
            references,
            hypotheses,
            weights=weights,
            smoothing_function=smoother,
        )
        scores[name] = round(float(score), 4)

    logger.info(
        f"Corpus BLEU | "
        f"B1: {scores['bleu1']:.4f} | "
        f"B2: {scores['bleu2']:.4f} | "
        f"B3: {scores['bleu3']:.4f} | "
        f"B4: {scores['bleu4']:.4f}"
    )
    return scores


def compute_sentence_bleu(
    reference_tokens: List[List[str]],
    hypothesis_tokens: List[str],
) -> Dict[str, float]:
    """
    Sentence-level BLEU for a single (reference, hypothesis) pair.

    Args:
        reference_tokens: List of reference token lists
        hypothesis_tokens: Single hypothesis token list

    Returns:
        Dict with bleu1-bleu4
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoother = SmoothingFunction().method1

    return {
        "bleu1": round(sentence_bleu(reference_tokens, hypothesis_tokens, (1, 0, 0, 0), smoother), 4),
        "bleu2": round(sentence_bleu(reference_tokens, hypothesis_tokens, (.5, .5, 0, 0), smoother), 4),
        "bleu3": round(sentence_bleu(reference_tokens, hypothesis_tokens, (1/3,)*3+(0,), smoother), 4),
        "bleu4": round(sentence_bleu(reference_tokens, hypothesis_tokens, (.25,)*4, smoother), 4),
    }


def compute_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Per-class and overall metrics for emotion recognition.

    Args:
        y_true: (N,) integer class labels
        y_pred: (N,) integer predicted labels

    Returns:
        Dict with accuracy, per-class precision/recall/f1, macro avg
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }


class RunningAverageMeter:
    """Tracks running average and std of a scalar metric."""

    def __init__(self, name: str):
        self.name = name
        self._values: List[float] = []

    def update(self, value: float):
        self._values.append(float(value))

    @property
    def mean(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self._values)) if self._values else 0.0

    @property
    def last(self) -> float:
        return self._values[-1] if self._values else 0.0

    def reset(self):
        self._values.clear()

    def __repr__(self) -> str:
        return f"{self.name}: {self.mean:.4f} ± {self.std:.4f}"
