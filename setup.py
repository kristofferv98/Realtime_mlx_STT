from setuptools import setup, find_packages

setup(
    name="Realtime_mlx_STT",
    version="0.1.0",
    description="Real-time speech-to-text transcription library optimized for Apple Silicon",
    author="Kristoffer Vatnehol",
    author_email="kristoffer.vatnehol@gmail.com",
    url="https://github.com/kristofferv98/Realtime_mlx_STT",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "RealtimeSTT": ["warmup_audio.wav"],
    },
    install_requires=[
        "numpy>=1.20.0",
        "pyaudio>=0.2.11",
        "mlx>=0.0.4",
        "librosa>=0.9.2",
        "tiktoken>=0.3.0",
        "huggingface_hub>=0.15.1",
        "webrtcvad>=2.0.10",
        "scipy>=1.8.0",
        "soundfile>=0.10.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.1.0",
            "isort>=5.12.0",
            "colorama>=0.4.4",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Operating System :: MacOS",
    ],
)