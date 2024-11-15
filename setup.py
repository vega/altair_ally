from setuptools import setup, find_packages


setup(
    name='altair_ally',
    url='https://joelostblom.github.io/altair_ally',
    author='Joel Ostblom',
    author_email='joel.ostblom@protonmail.com',
    packages=find_packages(),
    install_requires=['altair', 'vega_datasets', 'pandas'],
    python_requires='>=3.6',
    license='BSD-3',
    version='0.1.1',
    description='''Altair Ally is a companion package to Altair, which provides shortcuts to create common plots for exploratory data analysis, particularly those involving visualization of an entire dataset.''',
    # Include readme in markdown format, GFM markdown style by default
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
