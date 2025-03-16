"""Setup configuration for the pussel-backend package."""

from setuptools import find_packages, setup

setup(
    name="pussel-backend",
    version="0.1.0",
    packages=find_packages(include=["app", "app.*"]),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pillow",
        "pydantic",
        "pydantic-settings",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "isort",
        ],
    },
)
