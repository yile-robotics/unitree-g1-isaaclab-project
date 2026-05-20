"""Installation script for the unitree_g1_isaaclab Isaac Lab extension."""

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="unitree_g1_isaaclab",
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["toml"],
    zip_safe=False,
)
