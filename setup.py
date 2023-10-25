from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="EquiPy",
    version="0.0.1",
    author="Agathe F, Suzie G, Francois H, Philipp R",
    author_email="nemecontactepas@gmail.com",
    description="A package to get fairness on your predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a19ferna/ssa",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn',
                      'matplotlib', 'pandas', 'statsmodels', 'seaborn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
