"""Installation script for the unitree_g1_stand Isaac Lab extension."""
#把你的 unitree_g1_stand IsaacLab extension 安装成一个 Python package，让 IsaacLab / Python 可以 import 你的任务、配置、assets
import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="unitree_g1_stand",
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
