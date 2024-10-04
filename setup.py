"""setup file"""

import os

from setuptools import setup

install_requires = os.getenv("PACKAGE_REQS", "").split()

user_deps = []
if os.path.isfile("user_deps.txt"):
    with open("user_deps.txt") as f:
        user_deps = f.read().splitlines()

setup(
    name="ideas-toolbox-dlc",
    python_requires=">=3.9",
    version="1.0.0",
    packages=[],
    description="",
    url="https://github.com/inscopix/ideas-toolbox-dlc",
    install_requires=install_requires + user_deps,
)
