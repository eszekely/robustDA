import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "robustDA",
    version = "0.0.1",
    description = "Robust detection and attribution",
    author = "My name",
    packages = ["robustDA"],
    long_description = read('README.md'),
    python_requires = '>=3.7',
)