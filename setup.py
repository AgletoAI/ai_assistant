
from setuptools import setup, find_packages

setup(
    name="ai_assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "flask",
        "python-dotenv",
        "requests",
        "numpy",
        "pandas",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
    ],
)
