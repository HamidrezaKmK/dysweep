from setuptools import setup, find_packages
from dysweep import __version__

with open("readme_pypi.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dysweep",
    packages=find_packages(include=["dysweep", "dysweep.*"]),
    version=__version__,
    license="Apache License 2.0",
    description="Use Weights and Biases Sweeps for Dynamic Configuration generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hamid Kamkari",
    author_email="hamidrezakamkari@gmail.com",
    url="https://github.com/HamidrezaKmK/dysweep",
    entry_points={
        'console_scripts' : [
            'dysweep_create = dysweep.console:create_sweep',
            'dysweep_run_resume = dysweep.console:run_resume_sweep'
        ]
    },
    keywords=[
        "dynamic configurations",
        "large scale experiments",
        "deep learning",
        "sweeps",
        "hyperparameter tuning",
        "lazy evaluation"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Education",
        "Programming Language :: Python :: Implementation",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
)
