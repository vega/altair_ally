# Introduction

*This package is in early development and its API might change.*

Altair Ally is a companion package to Altair,
which provides a few shortcuts to create common plots
for exploratory data analysis (EDA),
particularly those involving visualizing an entire dataset.
The goal is to encourage good EDA habits
by making it quick and simple to create these visualizations,
and to provide useful default interactions with the plots.

Thanks to the excellent API in Altair / Vega-Lite,
some configuration is possible
by using the built-in methods of the returned chart objects,
but the general philosophy is that highly customized plots
are better made in Altair directly.
Check out the examples section to get started
and the API reference for all available options.

The package name is a nod to ggally
which complements ggplot2 in a similar,
but much more extensive manner.

## Installation

```
pip install altair-ally
```

The development version requires Python 3.11 or newer and supports Altair 6.x
and pandas 1.5.3 through 3.x. Examples use the built-in `altair.datasets` loader.
See the [compatibility and migration guide](compatibility.md) for installation
from a checkout and changes since the published 0.1.1 release.

## Usage

See [the examples section](examples.ipynb).
