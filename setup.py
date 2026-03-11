from setuptools import setup, find_packages

setup(
    name="emotion-caption-project",
    version="1.0.0",
    description="Real-Time Emotion Detection & Image Captioning",
    author="Your Name",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.12.0",
        "opencv-python>=4.8.0",
        "numpy>=1.23.0",
        "pandas>=2.0.0",
        "nltk>=3.8.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0",
        "fastapi>=0.103.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.4.0",
    ],
    entry_points={
        "console_scripts": [
            "train-emotion=scripts.train_emotion:main",
            "train-captioning=scripts.train_captioning:main",
            "export-model=scripts.export_model:main",
            "run-pipeline=inference.real_time_pipeline:main",
        ]
    },
)
