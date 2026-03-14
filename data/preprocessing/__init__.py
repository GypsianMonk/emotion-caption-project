"""Dataset preprocessors for FER-2013 and MS-COCO."""


def __getattr__(name):
    if name == "FER2013Preprocessor":
        from data.preprocessing.fer2013_preprocessor import FER2013Preprocessor  # noqa: F401
        return FER2013Preprocessor
    if name in ("COCOPreprocessor", "COCOVocabulary"):
        from data.preprocessing.coco_preprocessor import COCOPreprocessor, COCOVocabulary  # noqa: F401
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
