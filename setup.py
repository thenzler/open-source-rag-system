#!/usr/bin/env python3
"""
Setup script for Open Source RAG System
This ensures proper package installation and import paths
"""

from setuptools import setup, find_packages

setup(
    name="open-source-rag-system",
    version="1.0.0",
    description="Open Source RAG System with Ollama LLM Integration",
    packages=find_packages(include=["core", "core.*", "config", "config.*"]),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "requests>=2.32.3",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
    ],
    package_data={
        "core": ["**/*.py"],
        "config": ["**/*.py", "**/*.yaml", "**/*.yml"],
    },
    include_package_data=True,
    zip_safe=False,
)