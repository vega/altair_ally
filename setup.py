from pathlib import Path
from runpy import run_path

from setuptools import setup, find_packages

ROOT = Path(__file__).parent
VERSION = run_path(str(ROOT / 'altair_ally' / '_version.py'))['__version__']

setup(
    name='altair_ally',
    url='https://altair-viz.github.io/altair_ally/',
    author='Joel Ostblom',
    author_email='joel.ostblom@protonmail.com',
    packages=find_packages(),
    install_requires=[
        'altair>=6.0,<7',
        'pandas>=1.5.3,<4',
        'numpy>=1.23.5',
    ],
    extras_require={
        'test': ['pytest>=7', 'vl-convert-python>=1.9'],
        'doc': ['jupyter-book>=1,<2', 'session-info'],
    },
    python_requires='>=3.11',
    license='BSD-3',
    version=VERSION,
    description=(
        'Altair Ally is a companion package to Altair, which provides shortcuts '
        'to create common plots for exploratory data analysis, particularly '
        'those involving visualization of an entire dataset.'
    ),
    # Include readme in markdown format, GFM markdown style by default
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown'
)
