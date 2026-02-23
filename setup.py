"""setup file"""

import os

from setuptools import setup

install_requires = []
if os.path.isfile("user_deps.txt"):
    with open("user_deps.txt") as f:
        install_requires = f.read().splitlines()

setup(
    name="ideas-toolbox-dlc",
    python_requires=">=3.9",
    version="1.0.0",
    packages=[],
    description="",
    url="https://github.com/inscopix/ideas-toolbox-dlc",
    install_requires=[
        "isx==2.0.1",
        "ideas-python==1.1.1",
        "pytest==7.4.2",
    ]
    + install_requires,
)
