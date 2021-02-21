from os import path
from setuptools import setup, find_packages


setup(
    name='altair_ally',
    url='https://gitlab.com/joelostblom/sinfo',
    author='Joel Ostblom',
    author_email='joel.ostblom@protonmail.com',
    packages=find_packages(),
    install_requires=['altair', 'vega_datasets'],
    python_requires='>=3.6',
    license='BSD-3',
    description='''
        sinfo outputs version information for modules loaded in the current
        session, Python, and the OS.''',
)
