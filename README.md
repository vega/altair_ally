# Altair Ally

*This package is in early development and its API might change without notice.*

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

The development version requires **Python 3.11 or newer** and supports
**Altair 6.x**, including 6.3.0, and pandas 1.5.3 through 3.x.
Examples use Altair's built-in dataset loader, `altair.datasets`.

To try the unreleased compatibility updates from this checkout, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```sh
uv sync --locked
uv run python
```

## Documentation

[This Jupyter Book contains the documentation with more info and examples.](https://altair-viz.github.io/altair_ally/)

See [the compatibility and migration guide](doc/compatibility.md) for the
expanded `dist()` options and changes since 0.1.1.

## Development

```sh
uv sync --locked --all-groups
uv run task test
uv run --group doc task doc-build
uv run task build
```

The test suite validates chart specifications and renders them with Vega-Lite.
CI covers the minimum dependencies, Altair 6.0 and 6.3, and pandas 3.

Project metadata, development dependencies, pytest configuration, and tasks live
in `pyproject.toml`. The committed `uv.lock` makes development reproducible;
`.python-version` selects Python 3.13 locally, while CI also checks Python 3.11.
See [CONTRIBUTING.md](CONTRIBUTING.md) for dependency updates and CI overrides.
