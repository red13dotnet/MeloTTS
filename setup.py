import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

cwd = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(cwd, 'requirements.txt'), encoding='utf-8') as f:
    reqs = [line.strip() for line in f if line.strip() and not line.startswith('#')]

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.run([sys.executable, '-m', 'unidic', 'download'], check=True)

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        subprocess.run([sys.executable, '-m', 'unidic', 'download'], check=True)

setup(
    name='melotts',
    version='0.1.2',
    python_requires='>=3.11',
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    package_data={
        '': ['*.txt', 'cmudict_*'],
    },
    entry_points={
        "console_scripts": [
            "melotts = melo.main:main",
            "melo = melo.main:main",
            "melo-ui = melo.app:main",
        ],
    },
)