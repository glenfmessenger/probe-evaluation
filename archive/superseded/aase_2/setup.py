"""
AASE - Activation-based AI Safety Enforcement

A lightweight safety layer for vLLM using activation probes.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aase",
    version="0.1.0",
    author="Glen Messenger",
    description="Activation-based AI Safety Enforcement for vLLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "vllm>=0.4.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
        ],
    },
    include_package_data=True,
    package_data={
        "aase": ["pretrained/**/*.npy", "pretrained/**/*.json"],
    },
)
