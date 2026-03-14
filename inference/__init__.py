"""Inference modules for real-time emotion detection and caption generation."""


def __getattr__(name):
    if name == "EmotionDetector" or name == "FaceDetector":
        from inference.emotion_detector import EmotionDetector, FaceDetector  # noqa: F401
        return locals()[name]
    if name == "CaptionGenerator":
        from inference.caption_generator import CaptionGenerator  # noqa: F401
        return CaptionGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
