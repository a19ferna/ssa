from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ssa-a19ferna",
    version="0.0.1",
    author="Agathe F",
    author_email="agatheF@EMAIL.COM",
    description="A small package to work with prime numbers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a19ferna/ssa",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas', 'statsmodels', 'seaborn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)