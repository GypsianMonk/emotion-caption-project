"""Emotion recognition models."""


def __getattr__(name):
    if name in ("build_emotion_cnn", "compile_emotion_model", "model_summary_info"):
        from models.emotion.cnn_model import (  # noqa: F401
            build_emotion_cnn, compile_emotion_model, model_summary_info,
        )
        return locals()[name]
    if name == "EmotionTrainer":
        from models.emotion.trainer import EmotionTrainer  # noqa: F401
        return EmotionTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
