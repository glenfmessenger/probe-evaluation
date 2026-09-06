"""
AASE - Activation-based AI Safety Enforcement

A unified package for activation-based safety probes.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aase",
    version="0.1.0",
    author="Glen Messenger",
    author_email="glen@example.com",
    description="Activation-based AI Safety Enforcement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/aase",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "vllm": [
            "vllm>=0.4.0",
            "torch>=2.0.0",
        ],
        "transformers": [
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "accelerate>=0.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "scikit-learn>=1.0.0",
        ],
        "all": [
            "vllm>=0.4.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "accelerate>=0.20.0",
            "scikit-learn>=1.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "aase": ["pretrained/**/*.npy", "pretrained/**/*.json"],
    },
)
